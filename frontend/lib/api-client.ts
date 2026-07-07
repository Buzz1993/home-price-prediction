// Thin fetch wrapper around the existing FastAPI backend.
// Keeps all network concerns (base URL, JSON, auth token, error shape) in one
// place so service functions stay small and business logic remains in the backend.

// Defaults to the EstateMind Copilot API (port 8001). The Next.js app talks to
// this orchestration layer, not the ML Prediction API on 8000.
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

type RequestOptions = Omit<RequestInit, "body"> & {
  body?: unknown;
  token?: string | null;
};

// Attempts to pull a human-readable message out of a FastAPI error response.
async function parseError(response: Response): Promise<string> {
  try {
    const data = await response.json();
    const detail = data?.detail ?? data?.message;
    if (typeof detail === "string") return detail;
    if (detail?.error) return detail.error;
  } catch {
    // Non-JSON error body — fall through to the status text.
  }
  return response.statusText || "Request failed";
}

export async function apiRequest<T>(
  path: string,
  { body, token, headers, ...init }: RequestOptions = {}
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...headers,
    },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });

  if (!response.ok) {
    throw new ApiError(await parseError(response), response.status);
  }

  // 204 No Content and empty bodies decode to undefined.
  const text = await response.text();
  return (text ? JSON.parse(text) : undefined) as T;
}
