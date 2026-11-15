import { useCallback, useEffect, useRef, useState } from "react";
import { errorMessage } from "../services/api";

/** Run an async loader on mount / when deps change; exposes data, loading, error and reload. */
export function useAsync(loader, deps = []) {
  const [state, setState] = useState({ data: null, loading: true, error: null });
  const requestId = useRef(0);

  const run = useCallback(() => {
    const id = ++requestId.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    loader()
      .then((data) => id === requestId.current && setState({ data, loading: false, error: null }))
      .catch((err) => id === requestId.current && setState((s) => ({ ...s, loading: false, error: errorMessage(err) })));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(run, [run]);
  return { ...state, reload: run };
}
