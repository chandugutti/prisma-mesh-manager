"""
Prisma SD-WAN Mesh Manager — backend

Verified API URL format (prisma_sase Python SDK, May 2026):
  https://api.sase.paloaltonetworks.com/sdwan/{version}/api/{resource}

  NO /tenants/{id} in the path — the SASE gateway injects the tenant from
  the Bearer token scope (tsg_id:{tenant_id}). Including tenant in the URL
  causes the gateway to double it.

Existing anynet links are discovered via the topology endpoint
  POST /sdwan/v3.6/api/topology  {"type":"basenet","nodes":["<site_id>"]}
  (per-site, same approach as prisma_configure_mesh reference implementation).
  The topology response returns links with fields source_wan_if_id,
  target_wan_if_id, and path_id (= anynetlink ID used for deletion).
"""
import uuid
import asyncio
import json
import itertools
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

app = FastAPI(title="Prisma SD-WAN Mesh Manager")

# In-memory session store (single-user POC)
sessions: Dict[str, dict] = {}

DEFAULT_AUTH_URL = "https://auth.apps.paloaltonetworks.com/oauth2/access_token"
DEFAULT_API_HOST = "https://api.sase.paloaltonetworks.com"

# Endpoint versions from the official prisma_sase Python SDK
ENDPOINTS = {
    "sites":              ("v4.13", "sites"),
    "servicebindingmaps": ("v2.1",  "servicebindingmaps"),
    "wannetworks":        ("v2.1",  "wannetworks"),
    "waninterfaces":      ("v2.10", "sites/{site_id}/waninterfaces"),
    "topology":           ("v3.6",  "topology"),
    "anynetlinks":        ("v4.0",  "anynetlinks"),
    "elements":           ("v3.2",  "elements"),
}

# VPN tunnel scale limits per ION model, from Field Sizing Guide (6.5.1 practical limits).
# Separate dicts for SPOKE (branch) and HUB (DC) roles — ION 3200 / 5200 differ by role.
_VPN_LIMITS_SPOKE: Dict[str, int] = {
    "ion 1200":   128,
    "ion 1200-s": 200,
    "ion 3200":   220,
    "ion 5200":   450,
    "ion 9200":  3000,
}
_VPN_LIMITS_HUB: Dict[str, int] = {
    "ion 3200":   550,
    "ion 3200h": 2000,
    "ion 5200":  2500,
    "ion 9000":  6000,
    "ion 9200":  5500,
}


def _vpn_limit(model_name: str, role: str) -> int:
    """Return the VPN tunnel limit for a model+role, or 0 if unknown."""
    key = (model_name or "").lower().strip()
    table = _VPN_LIMITS_HUB if role == "HUB" else _VPN_LIMITS_SPOKE
    return table.get(key, 0)


# ── Models ────────────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    tenant_id:     str
    client_id:     str
    client_secret: str
    auth_url:      str = DEFAULT_AUTH_URL
    api_host:      str = DEFAULT_API_HOST


class SessionRequest(BaseModel):
    session_id: str


class MeshPreviewRequest(BaseModel):
    session_id: str
    domain_ids: List[str]


class MeshApplyRequest(BaseModel):
    session_id: str
    links: List[Dict[str, Any]]


class MeshRemoveRequest(BaseModel):
    session_id: str
    anynet_ids: List[str]


# ── URL builder ───────────────────────────────────────────────────────────────

def build_url(session: dict, endpoint: str, **kwargs) -> str:
    """
    Build a Prisma SD-WAN SASE API URL.
    Format: {api_host}/sdwan/{version}/api/{path}
    The SASE gateway injects the tenant from the Bearer token — do NOT add it here.
    """
    version, path_tpl = ENDPOINTS[endpoint]
    path = path_tpl.format(**kwargs)
    host = session["api_host"].rstrip("/")
    return f"{host}/sdwan/{version}/api/{path}"


# ── Session helpers ───────────────────────────────────────────────────────────

def get_session(session_id: str) -> dict:
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=401, detail="Session expired — please reconnect.")
    return s


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


async def api_get(url: str, token: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers(token))
    if resp.status_code not in (200, 201):
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Prisma API [{resp.status_code}] → {url}\n{resp.text[:400]}",
        )
    return resp.json()


async def api_post(url: str, token: str, data: dict) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=_headers(token), json=data)
    if resp.status_code not in (200, 201):
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Prisma API [{resp.status_code}] → {url}\n{resp.text[:400]}",
        )
    return resp.json()


async def api_delete(url: str, token: str) -> bool:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.delete(url, headers=_headers(token))
    return resp.status_code in (200, 204)


# ── Topology helpers ──────────────────────────────────────────────────────────

def _normalize_topo_link(link: dict) -> dict:
    """
    Normalize topology link fields to consistent names.
    Topology uses source/target naming; we map to ep1/ep2 for consistency.
    path_id is the anynetlink ID used for DELETE operations.
    """
    src = link.get("source_wan_if_id") or link.get("source_wan_path_id", "")
    dst = link.get("target_wan_if_id") or link.get("target_wan_path_id", "")
    return {
        "id":            link.get("path_id", ""),
        "ep1_wan_if_id": src,
        "ep2_wan_if_id": dst,
        "sub_type":      link.get("sub_type", ""),
        "path_status":   link.get("path_status", ""),
        "wan_type":      link.get("wan_type", ""),
        "source_site_id": link.get("source_node_id", ""),
        "target_site_id": link.get("target_node_id", ""),
    }


async def _fetch_site_topology(session: dict, site_id: str) -> List[dict]:
    """Query topology for a single site. Returns raw links list."""
    url = build_url(session, "topology")
    try:
        data = await api_post(url, session["token"], {"type": "basenet", "nodes": [site_id]})
        return data.get("links", [])
    except Exception:
        return []


async def _all_topology_links(session: dict, site_ids: List[str]) -> List[dict]:
    """
    Fetch topology for all given sites in parallel, deduplicate by path_id,
    and return normalized link objects.
    """
    results = await asyncio.gather(*[_fetch_site_topology(session, sid) for sid in site_ids])
    seen: set = set()
    links = []
    for site_links in results:
        for link in site_links:
            pid = link.get("path_id")
            if pid and pid not in seen:
                seen.add(pid)
                links.append(_normalize_topo_link(link))
    return links


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.post("/api/authenticate")
async def authenticate(req: AuthRequest):
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            req.auth_url,
            data={
                "grant_type":    "client_credentials",
                "client_id":     req.client_id,
                "client_secret": req.client_secret,
                "scope":         f"tsg_id:{req.tenant_id} email profile",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed [{resp.status_code}]: {resp.text[:400]}",
        )
    token = resp.json().get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="No access_token in auth response")

    host = req.api_host.rstrip("/")

    # SDK step 2: call profile to verify SD-WAN access and get the internal tenant_id
    async with httpx.AsyncClient(timeout=30) as client:
        prof_resp = await client.get(
            f"{host}/sdwan/v2.1/api/profile",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
    if prof_resp.status_code != 200:
        raise HTTPException(
            status_code=prof_resp.status_code,
            detail=(
                f"SD-WAN profile check failed [{prof_resp.status_code}]. "
                "Verify the service account has an SD-WAN role assigned in Prisma SASE IAM. "
                f"Detail: {prof_resp.text[:300]}"
            ),
        )
    profile = prof_resp.json()
    # Use the internal tenant_id from profile (may differ from TSG ID)
    internal_tenant_id = profile.get("tenant_id") or req.tenant_id

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "token":     token,
        "tenant_id": internal_tenant_id,
        "api_host":  host,
    }
    return {"session_id": session_id, "tenant_id": internal_tenant_id}


# ── Debug ─────────────────────────────────────────────────────────────────────

@app.post("/api/debug")
async def debug_urls(req: SessionRequest):
    s = get_session(req.session_id)
    return {
        "api_host":  s["api_host"],
        "tenant_id": s["tenant_id"],
        "urls": {
            "sites":             build_url(s, "sites"),
            "servicebindingmaps": build_url(s, "servicebindingmaps"),
            "wannetworks":       build_url(s, "wannetworks"),
            "topology":          build_url(s, "topology"),
            "anynetlinks_create": build_url(s, "anynetlinks"),
            "anynetlinks_delete": build_url(s, "anynetlinks") + "/<id>",
            "waninterfaces_ex":  build_url(s, "waninterfaces", site_id="<site_id>"),
        },
    }


# ── Data endpoints ────────────────────────────────────────────────────────────

@app.post("/api/sites")
async def get_sites(req: SessionRequest):
    s = get_session(req.session_id)
    data = await api_get(build_url(s, "sites"), s["token"])
    return data.get("items", [])


@app.post("/api/domains")
async def get_domains(req: SessionRequest):
    s = get_session(req.session_id)
    data = await api_get(build_url(s, "servicebindingmaps"), s["token"])
    return data.get("items", [])


@app.post("/api/wannetworks")
async def get_wan_networks(req: SessionRequest):
    s = get_session(req.session_id)
    data = await api_get(build_url(s, "wannetworks"), s["token"])
    return data.get("items", [])


@app.post("/api/topology")
async def get_topology(req: SessionRequest):
    """
    Fetch all existing anynet links by querying topology per site.
    Uses POST /sdwan/v3.6/api/topology {"type":"basenet","nodes":["<site_id>"]}
    per site, aggregated and deduplicated by path_id.
    """
    s = get_session(req.session_id)
    sites_data = await api_get(build_url(s, "sites"), s["token"])
    site_ids = [site["id"] for site in sites_data.get("items", []) if site.get("id")]
    return await _all_topology_links(s, site_ids)


# ── Mesh preview ──────────────────────────────────────────────────────────────

@app.post("/api/mesh/preview")
async def preview_mesh(req: MeshPreviewRequest):
    s     = get_session(req.session_id)
    token = s["token"]

    sites_raw, wn_raw, domains_raw, elements_raw = await asyncio.gather(
        api_get(build_url(s, "sites"),              token),
        api_get(build_url(s, "wannetworks"),         token),
        api_get(build_url(s, "servicebindingmaps"),  token),
        api_get(build_url(s, "elements"),            token),
    )

    sites        = sites_raw.get("items", [])
    wan_networks = {wn["id"]: wn for wn in wn_raw.get("items", [])}
    domains_map  = {d["id"]: d.get("name", d["id"]) for d in domains_raw.get("items", [])}

    # site_id → list of ION model names (multi-ION sites: use lowest limit)
    site_models: Dict[str, List[str]] = {}
    for elem in elements_raw.get("items", []):
        sid   = elem.get("site_id")
        model = elem.get("model_name", "")
        if sid and model:
            site_models.setdefault(sid, []).append(model)

    # Only SPOKE (branch) sites in the selected domains
    selected_sites = [
        site for site in sites
        if site.get("element_cluster_role") == "SPOKE"
        and site.get("service_binding") in req.domain_ids
    ]

    if not selected_sites:
        return {"new_links": [], "existing_count": 0, "summary": {
            "total_new": 0, "public_wan": 0, "private_wan": 0,
            "sites_included": 0, "domains_included": len(req.domain_ids),
        }}

    site_ids = [site["id"] for site in selected_sites]

    # Fetch WAN interfaces AND topology per site in parallel
    async def fetch_wanifs(site_id: str):
        try:
            url  = build_url(s, "waninterfaces", site_id=site_id)
            data = await api_get(url, token)
            return site_id, data.get("items", [])
        except Exception:
            return site_id, []

    wanif_results, topo_links = await asyncio.gather(
        asyncio.gather(*[fetch_wanifs(sid) for sid in site_ids]),
        _all_topology_links(s, site_ids),
    )
    site_wanifs: Dict[str, List] = dict(wanif_results)

    # Build existing anynet pair set from topology (both directions)
    existing_pairs: set = set()
    for link in topo_links:
        s1 = link["ep1_wan_if_id"]
        s2 = link["ep2_wan_if_id"]
        if s1 and s2:
            existing_pairs.add((s1, s2))
            existing_pairs.add((s2, s1))

    site_name_map   = {site["id"]: site.get("name", site["id"]) for site in selected_sites}
    site_domain_map = {site["id"]: site.get("service_binding") for site in selected_sites}

    new_links = []
    skipped   = 0

    for site1_id, site2_id in itertools.combinations(site_ids, 2):
        for swi1 in site_wanifs.get(site1_id, []):
            for swi2 in site_wanifs.get(site2_id, []):
                wn1   = wan_networks.get(swi1.get("network_id", ""), {})
                wn2   = wan_networks.get(swi2.get("network_id", ""), {})
                type1 = wn1.get("type")
                type2 = wn2.get("type")
                if not type1 or type1 != type2:
                    continue
                s1_id = swi1["id"]
                s2_id = swi2["id"]
                if (s1_id, s2_id) in existing_pairs:
                    skipped += 1
                    continue
                new_links.append({
                    "ep1_site_id":     site1_id,
                    "ep1_site_name":   site_name_map.get(site1_id),
                    "ep1_domain":      domains_map.get(site_domain_map.get(site1_id), ""),
                    "ep1_wan_if_id":   s1_id,
                    "ep1_wan_if_name": swi1.get("name", s1_id),
                    "ep1_wan_network": wn1.get("name", ""),
                    "ep2_site_id":     site2_id,
                    "ep2_site_name":   site_name_map.get(site2_id),
                    "ep2_domain":      domains_map.get(site_domain_map.get(site2_id), ""),
                    "ep2_wan_if_id":   s2_id,
                    "ep2_wan_if_name": swi2.get("name", s2_id),
                    "ep2_wan_network": wn2.get("name", ""),
                    "wan_type":        type1,
                })

    pub  = sum(1 for lnk in new_links if lnk["wan_type"] == "publicwan")
    priv = sum(1 for lnk in new_links if lnk["wan_type"] == "privatewan")

    # ── Scale guardrails ──────────────────────────────────────────────────────
    site_role_map = {site["id"]: site.get("element_cluster_role", "SPOKE") for site in selected_sites}

    # Count existing topology links per selected site (each link counted for both endpoints)
    existing_per_site: Dict[str, int] = {sid: 0 for sid in site_ids}
    for link in topo_links:
        for key in ("source_site_id", "target_site_id"):
            sid = link.get(key)
            if sid in existing_per_site:
                existing_per_site[sid] += 1

    # Count new links per site
    new_per_site: Dict[str, int] = {sid: 0 for sid in site_ids}
    for link in new_links:
        for ep in ("ep1_site_id", "ep2_site_id"):
            sid = link.get(ep)
            if sid in new_per_site:
                new_per_site[sid] += 1

    warnings = []
    for sid in site_ids:
        role   = site_role_map.get(sid, "SPOKE")
        models = site_models.get(sid, [])
        limits = [_vpn_limit(m, role) for m in models]
        limits = [lim for lim in limits if lim > 0]
        if not limits:
            continue
        limit    = min(limits)
        existing = existing_per_site.get(sid, 0)
        new_ct   = new_per_site.get(sid, 0)
        total    = existing + new_ct
        if total > limit:
            warnings.append({
                "site_id":        sid,
                "site_name":      site_name_map.get(sid, sid),
                "model":          models[0] if len(models) == 1 else ", ".join(models),
                "role":           role,
                "existing_links": existing,
                "new_links":      new_ct,
                "total_links":    total,
                "limit":          limit,
            })

    return {
        "new_links":      new_links,
        "existing_count": skipped,
        "warnings":       warnings,
        "summary": {
            "total_new":        len(new_links),
            "public_wan":       pub,
            "private_wan":      priv,
            "sites_included":   len(site_ids),
            "domains_included": len(req.domain_ids),
        },
    }


# ── Mesh apply ────────────────────────────────────────────────────────────────

@app.post("/api/mesh/apply")
async def apply_mesh(req: MeshApplyRequest):
    s   = get_session(req.session_id)
    url = build_url(s, "anynetlinks")

    async def event_stream():
        total   = len(req.links)
        success = 0
        failed  = 0
        for i, link in enumerate(req.links):
            payload = {
                "name":              None,
                "description":       None,
                "tags":              None,
                "ep1_site_id":       link["ep1_site_id"],
                "ep2_site_id":       link["ep2_site_id"],
                "ep1_wan_if_id":     link["ep1_wan_if_id"],
                "ep2_wan_if_id":     link["ep2_wan_if_id"],
                "forced":            True,
                "admin_up":          True,
                "type":              None,
                "vpnlink_configuration": None,
            }
            label = f"{link['ep1_site_name']} ↔ {link['ep2_site_name']}"
            try:
                await api_post(url, s["token"], payload)
                success += 1
                status, detail = "ok", ""
            except Exception as e:
                failed += 1
                status, detail = "error", str(e)[:120]

            yield f"data: {json.dumps({'index': i+1, 'total': total, 'success': success, 'failed': failed, 'status': status, 'link': label, 'detail': detail})}\n\n"
            await asyncio.sleep(0.05)

        yield f"data: {json.dumps({'done': True, 'success': success, 'failed': failed})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Mesh remove ───────────────────────────────────────────────────────────────

@app.post("/api/mesh/remove")
async def remove_mesh(req: MeshRemoveRequest):
    s = get_session(req.session_id)

    async def event_stream():
        total   = len(req.anynet_ids)
        success = 0
        failed  = 0
        for i, anynet_id in enumerate(req.anynet_ids):
            url = build_url(s, "anynetlinks") + f"/{anynet_id}"
            try:
                ok      = await api_delete(url, s["token"])
                success += ok
                failed  += not ok
                status   = "ok" if ok else "error"
            except Exception:
                failed += 1
                status   = "error"
            yield f"data: {json.dumps({'index': i+1, 'total': total, 'success': success, 'failed': failed, 'status': status, 'detail': anynet_id})}\n\n"
            await asyncio.sleep(0.05)

        yield f"data: {json.dumps({'done': True, 'success': success, 'failed': failed})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Static / root ─────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    resp = FileResponse("static/index.html", media_type="text/html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp
