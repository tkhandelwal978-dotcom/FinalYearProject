import axios from "axios";

export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

const client = axios.create({ baseURL: API_URL, timeout: 60000 });

/** Turn a server-relative path (e.g. /media/uploads/x.jpg) into an absolute URL. */
export const assetUrl = (path) => (path ? `${API_URL}${path}` : null);

/** Extract a readable message from an axios error. */
export function errorMessage(error) {
  if (error?.response?.data?.detail) {
    const detail = error.response.data.detail;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  }
  if (error?.code === "ECONNABORTED") return "The request timed out.";
  if (error?.request && !error?.response) return `Cannot reach the backend at ${API_URL}. Is it running?`;
  return error?.message || "Unexpected error";
}

export const api = {
  health: () => client.get("/api/health").then((r) => r.data),
  analytics: (days = 14) => client.get("/api/analytics", { params: { days } }).then((r) => r.data),
  modelInfo: () => client.get("/api/model-info").then((r) => r.data),
  history: (params) => client.get("/api/history", { params }).then((r) => r.data),
  prediction: (id) => client.get(`/api/history/${id}`).then((r) => r.data),
  deletePrediction: (id) => client.delete(`/api/history/${id}`).then((r) => r.data),
  predict: (file, { gradcam = true, onUploadProgress } = {}) => {
    const form = new FormData();
    form.append("file", file);
    return client
      .post("/api/predict", form, { params: { gradcam }, onUploadProgress })
      .then((r) => r.data);
  },
  reportUrl: (id) => `${API_URL}/api/history/${id}/report`,
  exportUrl: (predictedClass) =>
    `${API_URL}/api/history/export${predictedClass ? `?predicted_class=${predictedClass}` : ""}`,
};
