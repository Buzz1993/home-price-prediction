import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { QueryProvider } from "@/components/providers/query-provider";
import { AuthProvider } from "@/components/providers/auth-provider";
import { ThemeProvider } from "@/components/providers/theme-provider";
import { TooltipProvider } from "@/components/ui/tooltip";
import { THEME_STORAGE_KEY, THEMES } from "@/lib/themes";

const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "EstateMind Copilot",
  description: "AI-powered real estate search, analysis, and comparison.",
};

// Applies the persisted theme as `data-theme` on <html> before hydration, so
// the page first paints in the user's chosen theme (no flash of the default).
// Estate Green is the default and needs no attribute. Kept in sync with
// THEME_STORAGE_KEY / THEMES from lib/themes.ts.
const themeBootScript = `try{var t=localStorage.getItem(${JSON.stringify(
  THEME_STORAGE_KEY,
)});if(${JSON.stringify(
  THEMES.map((theme) => theme.id).slice(1),
)}.indexOf(t)>-1)document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col">
        <script dangerouslySetInnerHTML={{ __html: themeBootScript }} />
        <ThemeProvider>
          <QueryProvider>
            <AuthProvider>
              <TooltipProvider delayDuration={400}>
                {children}
              </TooltipProvider>
            </AuthProvider>
          </QueryProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
