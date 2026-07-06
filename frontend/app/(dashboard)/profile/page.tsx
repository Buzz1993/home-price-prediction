import { ProfileWorkspace } from "@/features/profile/profile-workspace";

// Profile (Phase 11): the user's account page. Consumes the documented GET
// /profile, GET /chat-history and GET /reports endpoints and shows user
// information, generated reports, AI chat history and a logout action. No
// authentication or backend business logic lives here.
export default function ProfilePage() {
  return <ProfileWorkspace />;
}
