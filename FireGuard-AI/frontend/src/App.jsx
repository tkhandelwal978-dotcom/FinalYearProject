import { Navigate, Route, Routes } from "react-router-dom";
import AppLayout from "./layouts/AppLayout";
import Overview from "./pages/Overview";
import Detect from "./pages/Detect";
import HistoryPage from "./pages/History";
import ModelPerformance from "./pages/ModelPerformance";
import SettingsPage from "./pages/Settings";

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Overview />} />
        <Route path="detect" element={<Detect />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="model" element={<ModelPerformance />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
