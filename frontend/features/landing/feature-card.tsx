import type { LucideIcon } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// Single reusable feature card for the landing "AI Features" section.
// Presentational only — reused for every capability so the section never
// duplicates card markup.
export type FeatureCardProps = {
  icon: LucideIcon;
  title: string;
  description: string;
};

export function FeatureCard({ icon: Icon, title, description }: FeatureCardProps) {
  return (
    <Card className="h-full gap-4 transition-shadow hover:shadow-md">
      <CardHeader>
        <span className="flex size-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
          <Icon className="size-5" />
        </span>
        <CardTitle className="mt-3 text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="text-sm text-muted-foreground text-pretty">
        {description}
      </CardContent>
    </Card>
  );
}
