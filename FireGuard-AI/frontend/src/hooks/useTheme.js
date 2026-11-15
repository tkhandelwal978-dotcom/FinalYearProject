import { useCallback, useEffect, useState } from "react";

const KEY = "fireguard-theme";

function readTheme() {
  try {
    return localStorage.getItem(KEY) || "dark";
  } catch {
    return "dark";
  }
}

export function useTheme() {
  const [theme, setTheme] = useState(readTheme);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      /* storage unavailable: theme still applies for this session */
    }
  }, [theme]);

  const toggle = useCallback(() => setTheme((t) => (t === "dark" ? "light" : "dark")), []);
  return { theme, setTheme, toggle };
}
