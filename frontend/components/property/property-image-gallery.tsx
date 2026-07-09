"use client";

// Reusable Property Image Gallery (Phase 14.2).
//
// The single image component used across the application (Property Cards,
// Property Details, and any future property image display). It renders only the
// images returned by the backend `image_urls` field — it never invents URLs or
// uses placeholder images when real ones are available.
//
// Features: hero image, thumbnail strip, image counter, previous/next
// navigation, fullscreen viewer, keyboard navigation (while fullscreen),
// mobile swipe, lazy loading, loading skeleton, and a graceful fallback when an
// image fails to load. Presentational only — no business logic lives here.

import { useCallback, useEffect, useRef, useState } from "react";
import { Dialog as DialogPrimitive } from "radix-ui";
import { ChevronLeft, ChevronRight, Expand, ImageOff, X } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

type PropertyImageGalleryProps = {
  // Backend `image_urls`. May be empty, one, or many.
  images?: string[] | null;
  // Accessible label / alt text base for the property.
  alt?: string;
  // Extra classes for the hero container (e.g. aspect ratio overrides).
  className?: string;
};

// Minimum horizontal travel (px) before a touch is treated as a swipe.
const SWIPE_THRESHOLD = 40;

export function PropertyImageGallery({
  images,
  alt = "Property image",
  className,
}: PropertyImageGalleryProps) {
  const list = (images ?? []).filter(Boolean);
  const total = list.length;

  const [selected, setSelected] = useState(0);
  const [fullscreen, setFullscreen] = useState(false);

  // Derive a valid index so the gallery stays correct if the image list shrinks
  // — no effect / cascading render needed.
  const current = total === 0 ? 0 : Math.min(selected, total - 1);

  const goTo = useCallback(
    (index: number) => {
      if (total === 0) return;
      setSelected(((index % total) + total) % total); // wrap around
    },
    [total]
  );

  const next = useCallback(() => goTo(current + 1), [current, goTo]);
  const prev = useCallback(() => goTo(current - 1), [current, goTo]);

  // Empty state — no images from the backend. Show a single placeholder panel;
  // the counter, thumbnails and navigation are all hidden.
  if (total === 0) {
    return (
      <div
        className={cn(
          "flex aspect-[4/3] w-full items-center justify-center overflow-hidden rounded-xl border bg-muted text-muted-foreground",
          className
        )}
      >
        <ImageOff className="size-8" />
      </div>
    );
  }

  const hasMany = total > 1;

  return (
    <div className="space-y-2">
      {/* Hero image — always image_urls[current]; clicking opens fullscreen. */}
      <div
        className={cn(
          "group relative aspect-[4/3] w-full overflow-hidden rounded-xl border bg-muted shadow-sm",
          className
        )}
      >
        <SwipeArea onSwipeLeft={next} onSwipeRight={prev} className="size-full">
          <button
            type="button"
            onClick={() => setFullscreen(true)}
            aria-label="Open fullscreen viewer"
            className="size-full cursor-zoom-in outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
          >
            <GalleryImage
              src={list[current]}
              alt={`${alt} ${current + 1} of ${total}`}
              className="size-full object-cover"
            />
          </button>
        </SwipeArea>

        {/* Expand hint. */}
        <span className="pointer-events-none absolute right-2 top-2 rounded-md bg-background/80 p-1.5 text-muted-foreground opacity-0 backdrop-blur transition-opacity group-hover:opacity-100">
          <Expand className="size-4" />
        </span>

        {/* Image counter (e.g. 1 / 12). */}
        {hasMany && (
          <span className="pointer-events-none absolute bottom-2 right-2 rounded-md bg-black/60 px-2 py-0.5 text-xs font-medium tabular-nums text-white">
            {current + 1} / {total}
          </span>
        )}

        {/* Previous / next navigation. */}
        {hasMany && (
          <>
            <NavButton
              side="left"
              onClick={prev}
              className="absolute left-2 top-1/2 -translate-y-1/2"
            />
            <NavButton
              side="right"
              onClick={next}
              className="absolute right-2 top-1/2 -translate-y-1/2"
            />
          </>
        )}
      </div>

      {/* Thumbnail strip — horizontally scrollable, active thumb highlighted. */}
      {hasMany && (
        <ThumbnailStrip
          images={list}
          current={current}
          alt={alt}
          onSelect={goTo}
        />
      )}

      {/* Fullscreen viewer. */}
      <FullscreenViewer
        open={fullscreen}
        onOpenChange={setFullscreen}
        images={list}
        current={current}
        alt={alt}
        onPrev={prev}
        onNext={next}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------

// A single image with a loading skeleton and a graceful failure fallback. Reused
// by the hero, thumbnails and fullscreen viewer so image loading behaves
// consistently everywhere. Never shows a broken-image icon.
function GalleryImage({
  src,
  alt,
  className,
  onError,
}: {
  src: string;
  alt: string;
  className?: string;
  onError?: () => void;
}) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const [renderedSrc, setRenderedSrc] = useState(src);

  // Reset loading/failure state when the source changes (index navigation
  // reuses the element) — the React-recommended adjust-state-during-render
  // pattern, no effect required.
  if (src !== renderedSrc) {
    setRenderedSrc(src);
    setLoaded(false);
    setFailed(false);
  }

  if (failed) {
    return (
      <div
        className={cn(
          "flex items-center justify-center bg-muted text-muted-foreground",
          className
        )}
      >
        <ImageOff className="size-8" />
      </div>
    );
  }

  return (
    <>
      {!loaded && <Skeleton className={cn("absolute inset-0", className)} />}
      {/* Plain img keeps arbitrary external listing URLs working without
          per-domain Next image config; lazy loading defers offscreen images. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={alt}
        loading="lazy"
        decoding="async"
        onLoad={() => setLoaded(true)}
        onError={() => {
          setFailed(true);
          onError?.();
        }}
        className={cn(className, !loaded && "opacity-0")}
      />
    </>
  );
}

// ---------------------------------------------------------------------------

function NavButton({
  side,
  onClick,
  className,
}: {
  side: "left" | "right";
  onClick: () => void;
  className?: string;
}) {
  const Icon = side === "left" ? ChevronLeft : ChevronRight;
  return (
    <Button
      type="button"
      variant="secondary"
      size="icon-sm"
      aria-label={side === "left" ? "Previous image" : "Next image"}
      onClick={(event) => {
        event.stopPropagation();
        onClick();
      }}
      className={cn(
        "bg-background/80 backdrop-blur hover:bg-background",
        className
      )}
    >
      <Icon />
    </Button>
  );
}

// ---------------------------------------------------------------------------

function ThumbnailStrip({
  images,
  current,
  alt,
  onSelect,
}: {
  images: string[];
  current: number;
  alt: string;
  onSelect: (index: number) => void;
}) {
  const activeRef = useRef<HTMLButtonElement>(null);

  // Keep the selected thumbnail in view as the user navigates.
  useEffect(() => {
    activeRef.current?.scrollIntoView({
      block: "nearest",
      inline: "nearest",
    });
  }, [current]);

  return (
    <div className="flex gap-2 overflow-x-auto pb-1">
      {images.map((src, index) => {
        const active = index === current;
        return (
          <button
            key={`${src}-${index}`}
            ref={active ? activeRef : undefined}
            type="button"
            onClick={() => onSelect(index)}
            aria-label={`View image ${index + 1}`}
            aria-current={active}
            className={cn(
              "relative aspect-[4/3] h-16 shrink-0 overflow-hidden rounded-lg border-2 bg-muted transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
              active
                ? "border-primary"
                : "border-transparent opacity-70 hover:opacity-100"
            )}
          >
            <GalleryImage
              src={src}
              alt={`${alt} thumbnail ${index + 1}`}
              className="size-full object-cover"
            />
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------

function FullscreenViewer({
  open,
  onOpenChange,
  images,
  current,
  alt,
  onPrev,
  onNext,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  images: string[];
  current: number;
  alt: string;
  onPrev: () => void;
  onNext: () => void;
}) {
  const total = images.length;
  const hasMany = total > 1;

  // Keyboard navigation is only active while the fullscreen viewer is open.
  // Radix Dialog already handles Esc to close; we add arrow-key navigation.
  useEffect(() => {
    if (!open || !hasMany) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        onPrev();
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        onNext();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, hasMany, onPrev, onNext]);

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/90 data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0" />
        <DialogPrimitive.Content className="fixed inset-0 z-50 flex flex-col outline-none">
          <DialogPrimitive.Title className="sr-only">
            {alt} — image {current + 1} of {total}
          </DialogPrimitive.Title>

          {/* Top bar: counter + close. */}
          <div className="flex items-center justify-between p-3 text-white">
            {hasMany ? (
              <span className="rounded-md bg-white/10 px-2 py-0.5 text-sm font-medium tabular-nums">
                {current + 1} / {total}
              </span>
            ) : (
              <span />
            )}
            <DialogPrimitive.Close asChild>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                aria-label="Close fullscreen viewer"
                className="text-white hover:bg-white/10 hover:text-white"
              >
                <X />
              </Button>
            </DialogPrimitive.Close>
          </div>

          {/* Image area. Clicking the empty backdrop (not the image) closes. */}
          <div
            className="relative flex min-h-0 flex-1 items-center justify-center p-4"
            onClick={(event) => {
              if (event.target === event.currentTarget) onOpenChange(false);
            }}
          >
            <SwipeArea
              onSwipeLeft={onNext}
              onSwipeRight={onPrev}
              className="flex size-full items-center justify-center"
            >
              {/* Preserve image quality — contain, no cropping. */}
              <GalleryImage
                key={images[current]}
                src={images[current]}
                alt={`${alt} ${current + 1} of ${total}`}
                className="max-h-full max-w-full object-contain"
              />
            </SwipeArea>

            {hasMany && (
              <>
                <NavButton
                  side="left"
                  onClick={onPrev}
                  className="absolute left-3 top-1/2 -translate-y-1/2"
                />
                <NavButton
                  side="right"
                  onClick={onNext}
                  className="absolute right-3 top-1/2 -translate-y-1/2"
                />
              </>
            )}
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

// ---------------------------------------------------------------------------

// Wraps children with touch handlers that translate a horizontal swipe into a
// previous/next action. Vertical scrolls are ignored.
function SwipeArea({
  onSwipeLeft,
  onSwipeRight,
  className,
  children,
}: {
  onSwipeLeft: () => void;
  onSwipeRight: () => void;
  className?: string;
  children: React.ReactNode;
}) {
  const start = useRef<{ x: number; y: number } | null>(null);

  return (
    <div
      className={className}
      onTouchStart={(event) => {
        const touch = event.touches[0];
        start.current = { x: touch.clientX, y: touch.clientY };
      }}
      onTouchEnd={(event) => {
        if (!start.current) return;
        const touch = event.changedTouches[0];
        const dx = touch.clientX - start.current.x;
        const dy = touch.clientY - start.current.y;
        start.current = null;
        // Horizontal swipe only.
        if (Math.abs(dx) < SWIPE_THRESHOLD || Math.abs(dx) < Math.abs(dy)) {
          return;
        }
        if (dx < 0) onSwipeLeft();
        else onSwipeRight();
      }}
    >
      {children}
    </div>
  );
}
