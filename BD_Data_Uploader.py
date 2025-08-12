# bd_uploader_v3.py — UX-focused Streamlit app for BlueDolphin
import json
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="BlueDolphin Uploader v3", layout="wide")
st.title("BlueDolphin CSV/Excel Uploader")

# ---------------- Connection (sidebar) ----------------
with st.sidebar:
    st.header("Connection")
    region = st.selectbox("Region", ["EU", "US"], index=0)
    API_BASE = "https://public-api.eu.bluedolphin.app/v1" if region == "EU" else "https://public-api.us.bluedolphin.app/v1"
    tenant = st.text_input("Tenant", placeholder="yourtenant")
    api_key = st.text_input("x-api-key", type="password")

def hdr() -> Dict[str, str]:
    return {"tenant": tenant or "", "x-api-key": api_key or "", "Content-Type": "application/json"}

def get_json(url: str, params: Dict = None):
    r = requests.get(url, headers=hdr(), params=params or {}, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {url} failed ({r.status_code}): {r.text[:300]}")
    return r.json()

def post_json(url: str, body: Dict):
    r = requests.post(url, headers=hdr(), data=json.dumps(body), timeout=60)
    if r.status_code not in (200, 201, 202):
        raise RuntimeError(f"POST {url} failed ({r.status_code}): {r.text[:300]}")
    return r.json() if r.text else {}

def patch_json(url: str, body: Dict):
    r = requests.patch(url, headers=hdr(), data=json.dumps(body), timeout=60)
    if r.status_code not in (200, 204):
        raise RuntimeError(f"PATCH {url} failed ({r.status_code}): {r.text[:300]}")
    return r.json() if r.text else {}

@st.cache_data(show_spinner=False)
def list_workspaces():
    return get_json(f"{API_BASE}/workspaces")

@st.cache_data(show_spinner=False)
def list_object_definitions():
    data = get_json(f"{API_BASE}/object-definitions")
    return data.get("items", data)

def list_objects(workspace_id: str, object_def_id: str, take: int = 2000):
    data = get_json(f"{API_BASE}/objects", params={"workspace_id": workspace_id, "filter": object_def_id, "take": take})
    return data.get("items", data) or []

def get_object(obj_id: str):
    return get_json(f"{API_BASE}/objects/{obj_id}")

def get_object_definition(def_id: str):
    return get_json(f"{API_BASE}/object-definitions/{def_id}")

def get_questionnaire(q_id: str):
    return get_json(f"{API_BASE}/questionnaires/{q_id}")

def create_object(title: str, object_def_id: str, workspace_id: str):
    body = {"object_title": title, "object_type_id": object_def_id, "workspace_id": workspace_id}
    return post_json(f"{API_BASE}/objects", body)

def patch_object(obj_id: str, body: Dict):
    return patch_json(f"{API_BASE}/objects/{obj_id}", body)

# ---------------- Step 1: workspace & definition ----------------
st.header("1) Pick workspace & object definition")
if not (tenant and api_key):
    st.info("Enter **tenant** and **x-api-key** in the sidebar.")
    st.stop()

try:
    ws = list_workspaces()
    ws_map = {w["name"]: w["id"] for w in ws}
except Exception as e:
    st.error(e); st.stop()

workspace_name = st.selectbox("Workspace", sorted(ws_map.keys()))
workspace_id = ws_map[workspace_name]

try:
    obj_defs = list_object_definitions()
    od_map = {od.get("name", od.get("id")): od["id"] for od in obj_defs}
except Exception as e:
    st.error(e); st.stop()

objdef_label = st.selectbox("Object definition", sorted(od_map.keys()))
object_def_id = od_map[objdef_label]

# ---------------- Step 2: upload file ----------------
st.header("2) Upload CSV / Excel")
up = st.file_uploader("Choose file", type=["csv", "xlsx", "xls"])
if not up:
    st.stop()

try:
    df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
except Exception as e:
    st.error(f"Could not read file: {e}"); st.stop()
if df.empty:
    st.error("The uploaded file is empty."); st.stop()
df.columns = [str(c) for c in df.columns]

# ---------------- Step 3: mapping (new UX) ----------------
st.header("3) Mapping")

# Fetch definition + questionnaires
with st.spinner("Loading definition & questionnaires…"):
    definition = get_object_definition(object_def_id)
    bd_properties = [p["name"] for p in (definition.get("object_properties") or [])]
    related_boem = definition.get("related_boem") or []
    questionnaires = []
    for q in related_boem:
        try:
            questionnaires.append(get_questionnaire(q["id"]))
        except Exception:
            pass

# Build questionnaire options: label -> (q_id, field_id)
boem_field_options: Dict[str, Tuple[str, str]] = {}
for q in questionnaires:
    qname = q.get("name", q.get("id"))
    for f in q.get("fields", []):
        label = f"{qname} – {f.get('name')} ({f.get('field_type')})"
        boem_field_options[label] = (q["id"], f["id"])

# Keep dynamic rows in session
if "prop_rows" not in st.session_state:
    st.session_state.prop_rows = [{"bd": "(select)", "csv": "(select)"}]
if "boem_rows" not in st.session_state:
    st.session_state.boem_rows = [{"bd": "(select)", "csv": "(select)"}]

# Title row
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("**Object Title (required)**")
with c2:
    title_col = st.selectbox("CSV column for title", list(df.columns), key="map_title", label_visibility="collapsed")

# Object ID row
c3, c4 = st.columns([1, 1])
with c3:
    st.markdown("Object ID (optional)")
with c4:
    object_id_col = st.selectbox("CSV column for object_id", ["(none)"] + list(df.columns), key="map_id", label_visibility="collapsed")

st.divider()

# Object properties (dynamic rows)
st.subheader("Object properties (optional)")
for i in range(len(st.session_state.prop_rows)):
    cA, cB = st.columns([1, 1])
    with cA:
        st.session_state.prop_rows[i]["bd"] = st.selectbox(
            "Property", ["(select)"] + bd_properties, key=f"prop_bd_{i}")
    with cB:
        st.session_state.prop_rows[i]["csv"] = st.selectbox(
            "CSV column", ["(select)"] + list(df.columns), key=f"prop_csv_{i}")
# Auto-append an empty row once the last is configured
last_prop = st.session_state.prop_rows[-1]
if last_prop["bd"] != "(select)" and last_prop["csv"] != "(select)":
    st.session_state.prop_rows.append({"bd": "(select)", "csv": "(select)"})

st.divider()

# Questionnaires (dynamic rows)
st.subheader("Questionnaires (optional)")
boem_labels = ["(select)"] + list(boem_field_options.keys())
for i in range(len(st.session_state.boem_rows)):
    cA, cB = st.columns([1, 1])
    with cA:
        st.session_state.boem_rows[i]["bd"] = st.selectbox(
            "Questionnaire – Field", boem_labels, key=f"boem_bd_{i}")
    with cB:
        st.session_state.boem_rows[i]["csv"] = st.selectbox(
            "CSV column", ["(select)"] + list(df.columns), key=f"boem_csv_{i}")
last_boem = st.session_state.boem_rows[-1]
if last_boem["bd"] != "(select)" and last_boem["csv"] != "(select)":
    st.session_state.boem_rows.append({"bd": "(select)", "csv": "(select)"})

# Build maps from configured rows
prop_map = {
    r["bd"]: r["csv"]
    for r in st.session_state.prop_rows
    if r["bd"] != "(select)" and r["csv"] != "(select)"
}
boem_map: Dict[Tuple[str, str], str] = {}
for r in st.session_state.boem_rows:
    if r["bd"] != "(select)" and r["csv"] != "(select)":
        boem_map[boem_field_options[r["bd"]]] = r["csv"]

if not title_col:
    st.error("Select a CSV column for **Object Title**.")
    st.stop()

# ---------------- Step 4: compute preview ----------------
st.header("4) Preview")
with st.spinner("Retrieving existing objects…"):
    existing = list_objects(workspace_id, object_def_id)

by_id = {o["id"]: o for o in existing if "id" in o}
by_title = {}
for o in existing:
    t = o.get("object_title") or o.get("title")
    if t:
        by_title[str(t)] = o

def row_target(row) -> Dict:
    tgt = {
        "object_id": ("" if object_id_col == "(none)" else str(row.get(object_id_col, "")).strip()),
        "object_title": str(row.get(title_col, "")).strip(),
        "props": {},
        "boem": {}  # {q_id: {field_id:value}}
    }
    for bd_name, csv_col in prop_map.items():
        val = row.get(csv_col, "")
        if pd.notna(val):
            tgt["props"][bd_name] = str(val)
    for (q_id, f_id), csv_col in boem_map.items():
        val = row.get(csv_col, "")
        if pd.notna(val):
            tgt["boem"].setdefault(q_id, {})
            tgt["boem"][q_id][f_id] = str(val)
    return tgt

def get_existing_detail(obj_stub: Dict) -> Dict:
    try:
        return get_object(obj_stub["id"])
    except Exception:
        # fallback to list item shape
        return {
            "id": obj_stub["id"],
            "object_title": obj_stub.get("object_title") or obj_stub.get("title") or "",
            "object_properties": [],
            "boem": []
        }

def read_prop(existing_detail: Dict) -> Dict[str, str]:
    return {p["name"]: str(p.get("value", "")) for p in existing_detail.get("object_properties", [])}

def read_boem(existing_detail: Dict) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for q in existing_detail.get("boem", []):
        qid = q.get("id")
        if not qid: 
            continue
        out[qid] = {it["id"]: str(it.get("value", "")) for it in q.get("items", [])}
    return out

summary_rows, detail_rows = [], []

for idx, row in df.iterrows():
    tgt = row_target(row)
    if not tgt["object_title"]:
        continue

    match = None
    if tgt["object_id"]:
        match = by_id.get(tgt["object_id"])
    if match is None:
        match = by_title.get(tgt["object_title"])

    if match:
        detail = get_existing_detail(match)
        current_title = str(detail.get("object_title", ""))
        will_update = False

        # title diff
        title_same = (tgt["object_title"] == current_title)
        if not title_same:
            will_update = True
        detail_rows.append({
            "Row": idx + 1, "Object ID": detail["id"], "Field type": "Title", "Field": "object_title",
            "Current value": current_title, "New value": tgt["object_title"],
            "Change": "Same" if title_same else "Update"
        })

        # properties diff for selected ones
        curr_props = read_prop(detail)
        for pname, nval in tgt["props"].items():
            cval = curr_props.get(pname, "")
            same = (str(cval) == str(nval))
            if not same:
                will_update = True
            detail_rows.append({
                "Row": idx + 1, "Object ID": detail["id"], "Field type": "Property", "Field": pname,
                "Current value": cval, "New value": nval, "Change": "Same" if same else "Update"
            })

        # questionnaire diff for selected ones
        curr_boem = read_boem(detail)
        for qid, fields in tgt["boem"].items():
            for fid, nval in fields.items():
                cval = curr_boem.get(qid, {}).get(fid, "")
                same = (str(cval) == str(nval))
                if not same:
                    will_update = True
                # label for display
                label = next((lbl for lbl, pair in boem_field_options.items() if pair == (qid, fid)), f"{qid}:{fid}")
                detail_rows.append({
                    "Row": idx + 1, "Object ID": detail["id"], "Field type": "Question", "Field": label,
                    "Current value": cval, "New value": nval, "Change": "Same" if same else "Update"
                })

        summary_rows.append({
            "Row": idx + 1, "Action": "Update" if will_update else "No change",
            "Object ID": detail["id"], "Title": tgt["object_title"]
        })
    else:
        # new object
        summary_rows.append({
            "Row": idx + 1, "Action": "Create", "Object ID": "", "Title": tgt["object_title"]
        })
        # show all mapped values as "Update" (creating)
        detail_rows.append({
            "Row": idx + 1, "Object ID": "(new)", "Field type": "Title", "Field": "object_title",
            "Current value": "", "New value": tgt["object_title"], "Change": "Update"
        })
        for pname, nval in tgt["props"].items():
            detail_rows.append({
                "Row": idx + 1, "Object ID": "(new)", "Field type": "Property", "Field": pname,
                "Current value": "", "New value": nval, "Change": "Update"
            })
        for qid, fields in tgt["boem"].items():
            for fid, nval in fields.items():
                label = next((lbl for lbl, pair in boem_field_options.items() if pair == (qid, fid)), f"{qid}:{fid}")
                detail_rows.append({
                    "Row": idx + 1, "Object ID": "(new)", "Field type": "Question", "Field": label,
                    "Current value": "", "New value": nval, "Change": "Update"
                })

summary_df = pd.DataFrame(summary_rows).sort_values(["Row", "Action"])
detail_df = pd.DataFrame(detail_rows).sort_values(["Row", "Field type", "Field"])

# Style: green for Same, red for Update (on New value)
def _style_colors(row):
    if row.get("Change") == "Same":
        return ["background-color: #e9f9ee" if c == "New value" else "" for c in row.index]
    else:
        return ["background-color: #ffecec" if c == "New value" else "" for c in row.index]

st.subheader("Summary")
st.dataframe(summary_df, use_container_width=True)

st.subheader("Details")
styled = detail_df.style.apply(_style_colors, axis=1)
st.dataframe(styled, use_container_width=True)

# ---------------- Step 5: apply ----------------
st.header("5) Apply changes")
apply = st.button("Apply now")
if apply:
    created = updated = skipped = errors = 0
    logs = []
    prog = st.progress(0.0, text="Working…")

    # Build quick lookup for detail rows per CSV row
    detail_by_row: Dict[int, pd.DataFrame] = {}
    for rnum, df_part in detail_df.groupby("Row"):
        detail_by_row[int(rnum)] = df_part

    for i, s in summary_df.iterrows():
        rnum = int(s["Row"])
        try:
            if s["Action"] == "No change":
                skipped += 1
                logs.append(f"Row {rnum}: no change")
            elif s["Action"] == "Create":
                part = detail_by_row.get(rnum, pd.DataFrame())
                title_val = part.loc[part["Field"] == "object_title", "New value"].iloc[0]
                res = create_object(title_val, object_def_id, workspace_id)
                new_id = res.get("id")
                # gather props & boem
                prop_payload = []
                boem_payload = {}
                for _, dr in part.iterrows():
                    if dr["Field type"] == "Property":
                        prop_payload.append({"name": dr["Field"], "value": dr["New value"]})
                    elif dr["Field type"] == "Question":
                        # lookup qid:fid by label
                        label = dr["Field"]
                        (qid, fid) = boem_field_options.get(label, (None, None))
                        if qid and fid:
                            boem_payload.setdefault(qid, []).append({"id": fid, "value": dr["New value"]})
                patch_body = {}
                if prop_payload:
                    patch_body["object_properties"] = prop_payload
                if boem_payload:
                    patch_body["boem"] = [{"id": qid, "items": items} for qid, items in boem_payload.items()]
                if patch_body:
                    patch_object(new_id, patch_body)
                created += 1
                logs.append(f"Row {rnum}: created id={new_id}")
            else:  # Update
                part = detail_by_row.get(rnum, pd.DataFrame())
                obj_id = str(s["Object ID"])
                patch_body = {}
                # title
                tdf = part[(part["Field type"] == "Title") & (part["Change"] == "Update")]
                if not tdf.empty:
                    patch_body["object_title"] = tdf["New value"].iloc[0]
                # props
                pdf = part[(part["Field type"] == "Property") & (part["Change"] == "Update")]
                if not pdf.empty:
                    patch_body["object_properties"] = [{"name": f, "value": v} for f, v in zip(pdf["Field"], pdf["New value"])]
                # boem
                qdf = part[(part["Field type"] == "Question") & (part["Change"] == "Update")]
                if not qdf.empty:
                    boem_payload = {}
                    for _, dr in qdf.iterrows():
                        (qid, fid) = boem_field_options.get(dr["Field"], (None, None))
                        if qid and fid:
                            boem_payload.setdefault(qid, []).append({"id": fid, "value": dr["New value"]})
                    if boem_payload:
                        patch_body["boem"] = [{"id": qid, "items": items} for qid, items in boem_payload.items()]
                if patch_body:
                    patch_object(obj_id, patch_body)
                    updated += 1
                    logs.append(f"Row {rnum}: updated id={obj_id}")
                else:
                    skipped += 1
                    logs.append(f"Row {rnum}: nothing to change for id={obj_id}")
        except Exception as e:
            errors += 1
            logs.append(f"Row {rnum}: ERROR {e}")
        prog.progress((len(logs)) / max(1, len(summary_df)))

    st.success(f"Done — Created: {created} • Updated: {updated} • No change: {skipped} • Errors: {errors}")
    st.code("\n".join(logs), language="text")

st.caption("Preview shows all mapped fields. Green = same, Red = will change. Nothing is written until you click Apply.")
