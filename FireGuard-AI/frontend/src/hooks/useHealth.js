import { useEffect, useState } from "react";
import { api, errorMessage } from "../services/api";

/** Polls /api/health so the shell can show live backend + model status. */
export function useHealth(intervalMs = 30000) {
  const [health, setHealth] = useState({ status: "checking", data: null, error: null });

  useEffect(() => {
    let cancelled = false;
    const check = () =>
      api
        .health()
        .then((data) => !cancelled && setHealth({ status: data.status, data, error: null }))
        .catch((err) => !cancelled && setHealth({ status: "offline", data: null, error: errorMessage(err) }));
    check();
    const timer = setInterval(check, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [intervalMs]);

  return health;
}
