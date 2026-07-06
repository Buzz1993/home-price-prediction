"use client";

// Property Details view (Phase 6). Displays the complete property record
// returned by GET /property/{id}: gallery, core information, amenities,
// features, location and price. Presentational only — all data comes from the
// backend. AI analysis sections (price, risk, rental, valuation) belong to a
// later phase and are not rendered here.

import Link from "next/link";
import {
  ArrowLeft,
  Building2,
  BedDouble,
  Image as ImageIcon,
  MapPin,
  Ruler,
  Sparkles,
} from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/ui/error-state";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { formatCr, splitList } from "@/features/dashboard/format";
import type { NotFoundResponse, PropertyDetail } from "@/types/dashboard";
import { usePropertyDetails } from "./use-property-details";

// Narrow the backend response, which is a property record or `{ error }`.
function isNotFound(
  data: PropertyDetail | NotFoundResponse
): data is NotFoundResponse {
  return "error" in data;
}

function BackLink() {
  return (
    <Button asChild variant="ghost" size="sm" className="-ml-2 w-fit">
      <Link href="/chat">
        <ArrowLeft /> Back to Copilot
      </Link>
    </Button>
  );
}

export function PropertyDetails({ id }: { id: string }) {
  const { data, isLoading, isError, isFetching, refetch } =
    usePropertyDetails(id);

  if (isLoading) {
    return (
      <div className="space-y-4">
        <BackLink />
        <Skeleton className="h-56 rounded-xl" />
        <Skeleton className="h-40 rounded-xl" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="space-y-4">
        <BackLink />
        <ErrorState
          title="Could not load this property"
          description="Something went wrong while loading the property details. Please try again."
          onRetry={() => refetch()}
          retrying={isFetching}
        />
      </div>
    );
  }

  if (!data || isNotFound(data)) {
    return (
      <div className="space-y-4">
        <BackLink />
        <Card>
          <CardContent className="p-6 text-center">
            <p className="font-medium">Property not found.</p>
            <p className="mt-1 text-sm text-muted-foreground">
              No property matches id{" "}
              <span className="font-mono">{id}</span>.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return <PropertyDetailsContent property={data} />;
}

function PropertyDetailsContent({ property }: { property: PropertyDetail }) {
  const amenities = splitList(property.amenities_mcp);
  const features = splitList(property.features_mcp);

  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <BackLink />

      {/* Image gallery — the backend dataset carries no images yet, so we show a
          consistent placeholder panel. */}
      <div className="flex h-56 items-center justify-center rounded-xl border bg-muted/40 text-muted-foreground">
        <ImageIcon className="size-10" />
      </div>

      {/* Header: name, location, price and BHK. */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 space-y-1">
          <h1 className="font-heading text-2xl font-semibold">
            {property.project_name || "Property"}
          </h1>
          <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
            <MapPin className="size-4 shrink-0" />
            <span className="truncate">{property.location}</span>
          </p>
          <p className="font-mono text-xs text-muted-foreground">
            {property.id}
          </p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-semibold">{formatCr(property.price)}</p>
          <Badge variant="secondary" className="mt-1">
            <BedDouble /> {property.bhk_type}
          </Badge>
        </div>
      </div>

      {/* Property information. */}
      <Card>
        <CardContent className="grid gap-4 p-6 sm:grid-cols-3">
          <InfoItem
            icon={<BedDouble className="size-4" />}
            label="Configuration"
            value={property.bhk_type}
          />
          <InfoItem
            icon={<Ruler className="size-4" />}
            label="Area"
            value={property.area ? `${property.area} sq.ft.` : "—"}
          />
          <InfoItem
            icon={<Building2 className="size-4" />}
            label="Builder"
            value={property.builder || "—"}
          />
        </CardContent>
      </Card>

      {/* Amenities. */}
      {amenities.length > 0 && (
        <Section title="Amenities">
          <div className="flex flex-wrap gap-1.5">
            {amenities.map((a) => (
              <Badge key={a} variant="outline">
                {a}
              </Badge>
            ))}
          </div>
        </Section>
      )}

      {/* Features. */}
      {features.length > 0 && (
        <Section title="Features">
          <div className="flex flex-wrap gap-1.5">
            {features.map((f) => (
              <Badge key={f} variant="secondary">
                {f}
              </Badge>
            ))}
          </div>
        </Section>
      )}

      {/* Backend analysis note. */}
      {property.analysis_msg && (
        <Section title="Overview">
          <p className="flex gap-2 text-sm text-muted-foreground">
            <Sparkles className="mt-0.5 size-4 shrink-0 text-primary" />
            <span>{property.analysis_msg}</span>
          </p>
        </Section>
      )}
    </div>
  );
}

function InfoItem({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="space-y-1">
      <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
        {icon}
        {label}
      </p>
      <p className="font-medium">{value}</p>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardContent className="space-y-3 p-6">
        <h2 className="font-heading text-sm font-semibold">{title}</h2>
        <Separator />
        {children}
      </CardContent>
    </Card>
  );
}
