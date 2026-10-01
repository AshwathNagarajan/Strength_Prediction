import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api/client";
import { Field, Notice, Page, Panel, Spinner } from "../components/Common";
import { useApp } from "../context/AppContext";

export default function Prediction() {
  const navigate=useNavigate();
  const {health,schema,savePrediction}=useApp(); const [domain,setDomain]=useState({}); const [values,setValues]=useState({});
  const [result,setResult]=useState(null); const [history,setHistory]=useState([]); const [busy,setBusy]=useState(false); const [error,setError]=useState("");
  const defaults=(kind="median")=>Object.fromEntries(Object.entries(domain).map(([name,item])=>[name,item.kind==="numeric"?item[kind]:kind==="mean"?item.unique_values.at(-1):item.unique_values[0]]));
  const loadHistory=()=>api.get("/prediction/history").then(response=>setHistory(response.data)).catch(()=>{});
  useEffect(()=>{api.get("/dataset/ranges").then(response=>{setDomain(response.data);setValues(Object.fromEntries(Object.entries(response.data).map(([name,item])=>[name,item.kind==="numeric"?item.median:item.unique_values[0]])));}).catch(()=>{});loadHistory();},[]);
  const run=async()=>{setBusy(true);setError("");try{const data=(await api.post("/predict",{features:values,explain:true})).data;setResult(data);savePrediction(data);loadHistory();}catch(e){setError(e.message);}finally{setBusy(false)}};
  const clearHistory=async()=>{await api.delete("/prediction/history");setHistory([])};
  const fields=schema?.columns?.filter(column=>column.role==="input")||[];
  const groups=["Material","Geometry","Connection","Other"].map(category=>({category,fields:fields.filter(field=>(field.category||"Other")===category)})).filter(group=>group.fields.length);
  return <Page title="Prediction" subtitle="Dynamic inputs are generated from the saved dataset schema and experimental ranges.">
    {!health?.model_trained&&<Notice type="warning">Train a model before prediction.</Notice>}{error&&<Notice type="error">{error}</Notice>}
    <Panel title="Beam and material inputs">{groups.map(group=><section key={group.category}><h3>{group.category} {group.category==="Connection"?"Properties":group.category==="Other"?"":"Properties"}</h3><div className="form-grid">{group.fields.map(field=>{const item=domain[field.name];return <Field key={field.name} label={field.display_name||field.name} unit={field.unit}>{item?.kind==="categorical"?<select value={values[field.name]??""} onChange={e=>setValues({...values,[field.name]:e.target.value})}>{item.unique_values.map(value=><option key={value}>{value}</option>)}</select>:<input type="number" min={item?.min} max={item?.max} step={field.step||"any"} value={values[field.name]??""} onChange={e=>setValues({...values,[field.name]:+e.target.value})}/>}</Field>})}</div></section>)}<div className="actions"><button onClick={()=>setValues(defaults("median"))}>Reset</button><button onClick={()=>setValues(defaults("mean"))}>Load example</button><button className="primary" disabled={busy||!health?.model_trained} onClick={run}>{busy?<Spinner label="Predicting..."/>:"Predict performance"}</button></div></Panel>
    {result&&<Panel title="Predicted structural performance"><p>Model used: <strong>{result.model_name}</strong></p><div className="metrics">{Object.entries(result.predictions).map(([name,value])=><div className="metric" key={name}><span>{name}</span><strong>{value.toFixed(3)}</strong></div>)}</div><Notice type={result.domain_analysis.overall==="IN_DOMAIN"?"success":"warning"}>Domain status: {result.domain_analysis.overall.replaceAll("_"," ")}</Notice><p>{result.natural_language_explanation}</p><p>{result.uncertainty?.message}</p><button onClick={()=>navigate("/explainability")}>Explain prediction</button></Panel>}
    {history.length>0&&<Panel title="Prediction history"><button onClick={clearHistory}>Clear history</button><div className="table-scroll"><table><thead><tr><th>Time</th><th>Model</th><th>Outputs</th></tr></thead><tbody>{history.slice().reverse().slice(0,20).map(row=><tr key={row.timestamp}><td>{new Date(row.timestamp).toLocaleString()}</td><td>{row.model}</td><td>{Object.entries(row.output).map(([name,value])=>`${name}: ${Number(value).toFixed(2)}`).join("; ")}</td></tr>)}</tbody></table></div></Panel>}
  </Page>;
}
