import React, { createContext, useContext, useEffect, useState } from "react";
import api from "../api/client";
const Context = createContext(null);
export function AppProvider({ children }) {
  const [health, setHealth] = useState(null);
  const [schema, setSchema] = useState(null);
  const [prediction, setPrediction] = useState(() => JSON.parse(localStorage.getItem("beam-prediction") || "null"));
  const [optimization, setOptimization] = useState(() => JSON.parse(localStorage.getItem("beam-optimization") || "null"));
  const refresh = async () => {
    setHealth((await api.get("/health")).data);
    try { setSchema((await api.get("/dataset/schema")).data); } catch { setSchema(null); }
  };
  useEffect(() => { refresh(); }, []);
  const savePrediction = (value) => { setPrediction(value); localStorage.setItem("beam-prediction", JSON.stringify(value)); };
  const saveOptimization = (value) => { setOptimization(value); localStorage.setItem("beam-optimization", JSON.stringify(value)); };
  return <Context.Provider value={{ health, schema, prediction, optimization, refresh, savePrediction, saveOptimization }}>{children}</Context.Provider>;
}
export const useApp = () => useContext(Context);
