import { PropertyDetails } from "@/features/property/property-details";

// Property Details route (Phase 6). Renders the full property record for the
// given id via the shared dashboard layout. Data flows from GET /property/{id}.
export default async function PropertyDetailsPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <PropertyDetails id={id} />;
}
