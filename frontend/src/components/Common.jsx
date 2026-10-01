import React from "react";
export function Page({ title, subtitle, actions, children }) { return <><div className="page-head"><div><div className="breadcrumb"><span>Workspace</span><b>/</b><strong>{title}</strong></div><h1>{title}</h1><p>{subtitle}</p></div>{actions&&<div className="page-actions">{actions}</div>}</div>{children}</>; }
export function Panel({ title, subtitle, children, className="", actions }) { return <section className={"panel " + className}><div className="panel-head"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{actions&&<div className="panel-actions">{actions}</div>}</div>{children}</section>; }
export function Notice({ type="info", children }) { return <div role={type==="error"?"alert":"status"} className={"notice " + type}>{children}</div>; }
export function Spinner({ label="Working..." }) { return <span className="spinner"><i/>{label}</span>; }
export function Metric({ label, value, hint }) { return <div className="metric"><span>{label}</span><strong>{value ?? "--"}</strong>{hint && <small>{hint}</small>}</div>; }
export function Field({ label, unit, children }) { return <label className="field"><span>{label}{unit && <em>{unit}</em>}</span>{children}</label>; }
export function StatusBadge({ tone="neutral", children }) { return <span className={`status-badge ${tone}`}>{children}</span>; }
