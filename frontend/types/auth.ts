// Shared authentication types. Mirrors the contract documented in
// project_docs/03_API.md (POST /signup, POST /login, GET /profile).

export type User = {
  id?: string;
  name: string;
  email: string;
};

export type LoginPayload = {
  email: string;
  password: string;
};

export type SignupPayload = {
  name: string;
  email: string;
  password: string;
};

// POST /login returns an authentication token. The user object is optional so
// the frontend can render immediately when the backend includes it.
export type AuthResponse = {
  token: string;
  user?: User;
};
