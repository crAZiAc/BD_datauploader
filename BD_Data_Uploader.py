# bd_uploader_v33.py — cache scoped to connection + live debug logging for GET/POST/PATCH (incl. 204)
import json
from typing import Dict, List, Tuple
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="BlueDolphin Uploader v33", layout="wide")
st.title("BlueDolphin CSV/Excel Uploader")

# ---------------- Sidebar: connection + debug ----------------
with st.sidebar:
    st.header("Connection")
    region = st.selectbox("Region", ["EU", "US"], index=0)
    API_BASE = "https://public-api.eu.bluedolphin.app/v1" if region == "EU" else "https://public-api.us.bluedolphin.app/v1"
    tenant = st.text_input("Tenant", placeholder="yourtenant")
    api_key = st.text_input("x-api-key", type="password")

    st.divider()
    debug_mode = st.checkbox("Debug mode (log API calls)", value=False)

    # debug storage + live box
    if "debug_log" not in st.session_state:
        st.session_state.debug_log = []
    if "debug_box" not in st.session_state:
        st.session_state.debug_box = st.empty()

    col_l, col_r = st.columns([1, 1])
    with col_l:
        if st.button("Clear log"):
            st.session_state.debug_log = []
    with col_r:
        st.caption(f"Log lines: {len(st.session_state.debug_log)}")

    st.session_state.debug_box.code("\n".join(st.session_state.debug_log) or "(empty)", language="text")

def _dbg(line: str):
    if not st.session_state.get("debug_log") is None and debug_mode:
        st.session_state.debug_log.append(line)
        box = st.session_state.get("debug_box")
        if box:
            box.code("\n".join(st.session_state.debug_log), language="text")

def hdr() -> Dict[str, str]:
    # never log secrets
    return {"tenant": tenant or "", "x-api-key": api_key or "", "Content-Type": "application/json"}

# Clear caches when connection changes (+ button)
conn_key = f"{region}|{tenant}|{(api_key or '')[:8]}"
if st.session_state.get("conn_key") != conn_key:
    st.cache_data.clear()
    st.session_state.conn_key = conn_key
    for k in ("preview_df", "preview_meta", "prop_rows", "boem_rows"):
        st.session_state.pop(k, None)
if st.sidebar.button("Reload data (clear cache)"):
    st.cache_data.clear()
    for k in ("preview_df", "preview_meta"):
        st.session_state.pop(k, None)

# -------- unified request with full logging (pre + post, incl. 204) --------
def _request(method: str, path: str, *, params: Dict=None, body: Dict=None, expect_json: bool=True):
    url = f"{API_BASE}{path}"

    # pre-call
    if debug_mode:
        body_keys = list(body.keys()) if isinstance(body, dict) else None
        prop_cnt = len(body.get("object_properties", [])) if isinstance(body, dict) and "object_properties" in body else 0
        boem_cnt = sum(len(q.get("items", [])) for q in body.get("boem", [])) if isinstance(body, dict) and "boem" in body else 0
        _dbg(f">>> {method} {path} {params or ''}"
             f"{(' body_keys=' + str(body_keys)) if body_keys else ''}"
             f"{(' props=' + str(prop_cnt)) if prop_cnt else ''}"
             f"{(' boem_items=' + str(boem_cnt)) if boem_cnt else ''}")

    r = requests.request(
        method, url,
        headers=hdr(),
        params=params or {},
        data=(json.dumps(body) if body is not None else None),
        timeout=60
    )

    status_line = f"[{r.status_code}] {method} {path}"

    if r.status_code < 400:
        # success (log even if empty 204)
        if debug_mode:
            if r.headers.get("Content-Type", "").startswith("application/json") and r.text:
                try:
                    payload = r.json()
                    if isinstance(payload, dict) and "items" in payload and isinstance(payload["items"], list):
                        _dbg(f"{status_line} items={len(payload['items'])}")
                    elif isinstance(payload, dict) and "id" in payload:
                        _dbg(f"{status_line} id={payload.get('id')}")
                    else:
                        _dbg(status_line)
                except Exception:
                    _dbg(status_line)
            else:
                _dbg(status_line)
        if expect_json and r.text:
            try:
                return r.json()
            except Exception:
                return r.text
        return r.text or ""

    # errors
    msg = (r.text or "").strip()
    st.error(f"{status_line}: {msg[:800]}")
    _dbg(f"{status_line} ERROR: {msg[:800]}")
    r.raise_for_status()

# Thin helpers
def get_json(path: str, params: Dict=None):
    return _request("GET", path, params=params, expect_json=True)

def post_json(path: str, body: Dict):
    return _request("POST", path, body=body, expect_json=True)

def patch_json(path: str, body: Dict):
    return _request("PATCH", path, body=body, expect_json=True)

# -------- cached fetchers keyed by connection --------
@st.cache_data(show_spinner=False)
def list_workspaces_cached(api_base: str, tenant_: str, api_key_: str):
    return get_json("/workspaces")

@st.cache_data(show_spinner=False)
def list_object_definitions_cached(api_base: str, tenant_: str, api_key_: str):
    data = get_json("/object-definitions")
    return data.get("items", data)

# Uncached (fresh each preview/apply)
def list_objects(workspace_id: str, object_def_id: str, take: int=2000):
    data = get_json("/objects", params={"workspace_id": workspace_id, "filter": object_def_id, "take": take})
    return data.get("items", data) or []

def get_object(obj_id: str):
    return get_json(f"/objects/{obj_id}")

def get_object_definition(def_id: str):
    return get_json(f"/object-definitions/{def_id}")

def get_questionnaire(q_id: str):
    return get_json(f"/questionnaires/{q_id}")

def create_object(title: str, object_def_id: str, workspace_id: str):
    body = {"object_title": title, "object_type_id": object_def_id, "workspace_id": workspace_id}
    return post_json("/objects", body)

def patch_object(obj_id: str, body: Dict):
    return patch_json(f"/objects/{obj_id}", body)

# ---------------- Step 1: pick workspace & object definition ----------------
st.header("1) Pick workspace & object definition")
if not (tenant and api_key):
    st.info("Enter **tenant** and **x-api-key** in the sidebar.")
    st.stop()

try:
    ws = list_workspaces_cached(API_BASE, tenant, api_key); ws_map = {w["name"]: w["id"] for w in ws}
except Exception as e:
    st.error(e); st.stop()
workspace = st.selectbox("Workspace", sorted(ws_map.keys()))
workspace_id = ws_map[workspace]

try:
    obj_defs = list_object_definitions_cached(API_BASE, tenant, api_key); od_map = {od.get("name", od.get("id")): od["id"] for od in obj_defs}
except Exception as e:
    st.error(e); st.stop()
objdef_label = st.selectbox("Object definition", sorted(od_map.keys()))
object_def_id = od_map[objdef_label]

# ---------------- Step 2: upload ----------------
st.header("2) Upload CSV / Excel")
up = st.file_uploader("Choose file", type=["csv", "xlsx", "xls"])
if not up: st.stop()
try:
    df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
except Exception as e:
    st.error(f"Could not read file: {e}"); st.stop()
if df.empty:
    st.error("The uploaded file is empty."); st.stop()
df.columns = [str(c) for c in df.columns]

# ---------------- Step 3: mapping (UX from v3.1) ----------------
st.header("3) Mapping")
with st.spinner("Loading definition & questionnaires…"):
    definition = get_object_definition(object_def_id)
    bd_properties = [p["name"] for p in (definition.get("object_properties") or [])]
    related_boem = definition.get("related_boem") or []
    questionnaires = []
    for q in related_boem:
        try: questionnaires.append(get_questionnaire(q["id"]))
        except Exception: pass

boem_field_options: Dict[str, Tuple[str, str, str, str]] = {}
for q in questionnaires:
    qname = q.get("name", q.get("id"))
    for f in q.get("fields", []):
        fname = f.get("name")
        label = f"{qname} – {fname}"
        boem_field_options[label] = (q["id"], f["id"], qname, fname)

if "prop_rows" not in st.session_state: st.session_state.prop_rows = [{"bd":"(select)","csv":"(select)"}]
if "boem_rows" not in st.session_state: st.session_state.boem_rows = [{"bd":"(select)","csv":"(select)"}]

# Title row
c1, c2 = st.columns([1, 1])
with c1: st.markdown("**Object Title (required)**")
with c2: title_col = st.selectbox("CSV column for title", list(df.columns), key="map_title", label_visibility="collapsed")

# Object ID row
c3, c4 = st.columns([1, 1])
with c3: st.markdown("Object ID (optional)")
with c4: object_id_col = st.selectbox("CSV column for object_id", ["(none)"] + list(df.columns), key="map_id", label_visibility="collapsed")

st.divider()
st.subheader("Object properties (optional)")
for i in range(len(st.session_state.prop_rows)):
    cA, cB = st.columns([1, 1])
    with cA:
        st.session_state.prop_rows[i]["bd"] = st.selectbox("Property", ["(select)"] + bd_properties, key=f"prop_bd_{i}")
    with cB:
        st.session_state.prop_rows[i]["csv"] = st.selectbox("CSV column", ["(select)"] + list(df.columns), key=f"prop_csv_{i}")
if st.session_state.prop_rows[-1]["bd"]!="(select)" and st.session_state.prop_rows[-1]["csv"]!="(select)":
    st.session_state.prop_rows.append({"bd":"(select)","csv":"(select)"})

st.divider()
st.subheader("Questionnaires (optional)")
boem_labels = ["(select)"] + list(boem_field_options.keys())
for i in range(len(st.session_state.boem_rows)):
    cA, cB = st.columns([1, 1])
    with cA:
        st.session_state.boem_rows[i]["bd"] = st.selectbox("Questionnaire – Field", boem_labels, key=f"boem_bd_{i}")
    with cB:
        st.session_state.boem_rows[i]["csv"] = st.selectbox("CSV column", ["(select)"] + list(df.columns), key=f"boem_csv_{i}")
if st.session_state.boem_rows[-1]["bd"]!="(select)" and st.session_state.boem_rows[-1]["csv"]!="(select)":
    st.session_state.boem_rows.append({"bd":"(select)","csv":"(select)"})

# Build maps
prop_map = {r["bd"]: r["csv"] for r in st.session_state.prop_rows if r["bd"]!="(select)" and r["csv"]!="(select)"}
boem_map: Dict[Tuple[str,str,str,str], str] = {}
for r in st.session_state.boem_rows:
    if r["bd"]!="(select)" and r["csv"]!="(select)":
        boem_map[boem_field_options[r["bd"]]] = r["csv"]

if not title_col:
    st.error("Select a CSV column for **Object Title**."); st.stop()

# ---------------- Step 4: Preview (Generate) ----------------
st.header("4) Preview")
if st.button("Generate preview"):
    with st.spinner("Retrieving existing objects…"):
        existing = list_objects(workspace_id, object_def_id)

    by_id = {o["id"]: o for o in existing if "id" in o}
    by_title = {}
    for o in existing:
        t = o.get("object_title") or o.get("title")
        if t: by_title[str(t)] = o

    def get_detail(stub):
        try: return get_object(stub["id"])
        except Exception:
            return {"id": stub["id"], "object_title": stub.get("object_title") or stub.get("title") or "", "object_properties": [], "boem": []}

    def read_props(detail): return {p["name"]: str(p.get("value","")) for p in detail.get("object_properties", [])}

    def read_boem(detail):
        out={}
        for q in detail.get("boem", []):
            qid=q.get("id"); 
            if not qid: continue
            out[qid]={it["id"]:str(it.get("value","")) for it in q.get("items", [])}
        return out

    # Columns: Action, Object_Title, Id, then mapped fields
    preview_cols = ["Action", "Object_Title", "Id"]
    prop_cols = [f"objectproperty_{p}" for p in prop_map.keys()]
    boem_cols = [f"questionnaire({qname})_{fname}" for (_,_,qname,fname) in boem_map.keys()]
    preview_cols += prop_cols + boem_cols

    rows, mask_rows, meta = [], [], []

    for _, r in df.iterrows():
        title_target = str(r.get(title_col, "")).strip()
        if not title_target:
            continue
        obj_id_val = "" if object_id_col=="(none)" else str(r.get(object_id_col, "")).strip()
        target_props = {p: ("" if pd.isna(r.get(csv,"")) else str(r.get(csv,""))) for p, csv in prop_map.items()}
        target_boem = {}
        for (qid,fid,qname,fname), csv in boem_map.items():
            val = "" if pd.isna(r.get(csv,"")) else str(r.get(csv,""))
            target_boem.setdefault(qid, {})[fid] = val

        stub = by_id.get(obj_id_val) if obj_id_val else None
        if stub is None:
            stub = by_title.get(title_target)

        if stub:
            detail = get_detail(stub)
            curr_title = str(detail.get("object_title",""))
            curr_props = read_props(detail)
            curr_boem  = read_boem(detail)

            row = {"Action":"Update","Object_Title":title_target,"Id":detail["id"]}
            mask = {"Action":False,"Object_Title":(title_target!=curr_title),"Id":False}
            any_change = (title_target!=curr_title)

            for p in prop_map.keys():
                newv = target_props.get(p,""); oldv = curr_props.get(p,"")
                row[f"objectproperty_{p}"] = newv
                chg = (str(newv)!=str(oldv))
                mask[f"objectproperty_{p}"] = chg; any_change = any_change or chg

            for (qid,fid,qname,fname) in {(q,f,qn,fn) for (q,f,qn,fn) in boem_map.keys()}:
                key = f"questionnaire({qname})_{fname}"
                newv = target_boem.get(qid, {}).get(fid, "")
                oldv = curr_boem.get(qid, {}).get(fid, ""
                )
                row[key] = newv
                chg = (str(newv)!=str(oldv))
                mask[key] = chg; any_change = any_change or chg

            if any_change:
                rows.append(row); mask_rows.append(mask)
                meta.append({
                    "new": False,
                    "id": detail["id"],
                    "title_update": title_target if (title_target!=curr_title) else None,
                    "prop_updates": {p: target_props[p] for p in prop_map.keys() if mask.get(f"objectproperty_{p}", False)},
                    "boem_updates": {qid: {fid: target_boem[qid][fid]
                                           for fid in target_boem.get(qid, {})
                                           if mask.get(f"questionnaire({[v[2] for v in boem_map.keys() if v[0]==qid and v[1]==fid][0]})_{[v[3] for v in boem_map.keys() if v[0]==qid and v[1]==fid][0]}", False)}
                                     for qid in target_boem.keys()}
                })
        else:
            row = {"Action":"Create","Object_Title":title_target,"Id":""}
            mask = {"Action":False,"Object_Title":True,"Id":False}
            for p in prop_map.keys():
                row[f"objectproperty_{p}"] = target_props.get(p, ""); mask[f"objectproperty_{p}"] = True
            for (qid,fid,qname,fname) in boem_map.keys():
                key = f"questionnaire({qname})_{fname}"
                row[key] = target_boem.get(qid, {}).get(fid, ""); mask[key] = True
            rows.append(row); mask_rows.append(mask)
            meta.append({"new": True, "id": "", "title": title_target, "props": target_props, "boem": target_boem})

    if not rows:
        st.info("Nothing to create or update based on current mapping."); st.stop()

    preview_df = pd.DataFrame(rows, columns=preview_cols)
    mask_df = pd.DataFrame(False, index=preview_df.index, columns=preview_df.columns)
    for i, m in enumerate(mask_rows):
        for k, v in m.items():
            if k in mask_df.columns:
                mask_df.loc[preview_df.index[i], k] = bool(v)

    def style_fn(val, col, idx):
        is_change = bool(mask_df.loc[idx, col]) if col in mask_df.columns else False
        if col in ("Action","Id"):  # no coloring
            return ""
        bg = "#ffecec" if is_change else "#e9f9ee"
        fg = "#7a0a0a" if is_change else "#0b5f2a"
        return f"background-color: {bg}; color: {fg};"

    styled = preview_df.style.apply(lambda s: [style_fn(v, s.name, s.index[i]) for i, v in enumerate(s)], axis=0)
    st.dataframe(styled, use_container_width=True)

    st.session_state.preview_df = preview_df
    st.session_state.preview_mask = mask_df
    st.session_state.preview_meta = meta
    st.success(f"Preview ready: {len(preview_df)} rows")

# ---------------- Step 5: Apply (uses preview) ----------------
st.header("5) Apply changes")
if st.button("Apply now", disabled=("preview_df" not in st.session_state)):
    if "preview_df" not in st.session_state:
        st.warning("Click **Generate preview** first."); st.stop()

    meta = st.session_state.preview_meta
    created = updated = errors = 0
    logs = []
    prog = st.progress(0.0, text="Working…")

    def boem_payload_from_dict(d: Dict[str, Dict[str,str]]):
        return [{"id": qid, "items": [{"id": fid, "value": val} for fid, val in fields.items()]} for qid, fields in d.items() if fields]

    for i, item in enumerate(meta):
        try:
            if item["new"]:
                res = create_object(item["title"], object_def_id, workspace_id)
                new_id = res.get("id")
                patch_body = {}
                if item["props"]:
                    patch_body["object_properties"] = [{"name": k, "value": v} for k, v in item["props"].items()]
                if item["boem"]:
                    patch_body["boem"] = boem_payload_from_dict(item["boem"])
                if patch_body:
                    patch_object(new_id, patch_body)
                created += 1
                logs.append(f"Create → id={new_id}")
            else:
                patch_body = {}
                if item["title_update"]:
                    patch_body["object_title"] = item["title_update"]
                if item["prop_updates"]:
                    patch_body["object_properties"] = [{"name": k, "value": v} for k, v in item["prop_updates"].items()]
                boem_clean = {qid: fields for qid, fields in (item.get("boem_updates") or {}).items() if fields}
                if boem_clean:
                    patch_body["boem"] = boem_payload_from_dict(boem_clean)
                if patch_body:
                    patch_object(item["id"], patch_body)
                    updated += 1
                    logs.append(f"Update → id={item['id']}")
        except Exception as e:
            errors += 1
            logs.append(f"ERROR: {e}")
        prog.progress((i+1)/max(1,len(meta)))

    st.success(f"Done — Created: {created} • Updated: {updated} • Errors: {errors}")
    st.code("\n".join(logs), language="text")

st.caption("Preview shows target values; green = unchanged, red = will change. Debug mode logs GET/POST/PATCH (incl. 204).")
