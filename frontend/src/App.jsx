import React, { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
const Dataset = lazy(() => import("./pages/Dataset"));
const Training = lazy(() => import("./pages/Training"));
const Prediction = lazy(() => import("./pages/Prediction"));
const Explainability = lazy(() => import("./pages/Explainability"));
const Optimization = lazy(() => import("./pages/Optimization"));
const Results = lazy(() => import("./pages/Results"));
export default function App() {
  return <Layout><Suspense fallback={<div className="notice">Loading workspace...</div>}><Routes>
    <Route path="/" element={<Dashboard />} /><Route path="/dataset" element={<Dataset />} />
    <Route path="/training" element={<Training />} /><Route path="/prediction" element={<Prediction />} />
    <Route path="/explainability" element={<Explainability />} /><Route path="/optimization" element={<Optimization />} />
    <Route path="/results" element={<Results />} /><Route path="*" element={<Navigate to="/" replace />} />
  </Routes></Suspense></Layout>;
}
