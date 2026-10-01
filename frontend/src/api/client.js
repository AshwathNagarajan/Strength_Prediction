import axios from "axios";
const api = axios.create({ baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000/api", timeout: 120000 });
api.interceptors.response.use((response) => {
  if (response.data?.success === true && Object.prototype.hasOwnProperty.call(response.data, "data")) response.data = response.data.data;
  return response;
}, (error) => Promise.reject(new Error(error.response?.data?.message || error.response?.data?.detail || error.message)));
export default api;
