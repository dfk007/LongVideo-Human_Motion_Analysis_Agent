import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Motion Analysis Agent",
  description: "AI-powered long-video analysis agent",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-slate-50 min-h-screen text-slate-900">{children}</body>
    </html>
  );
}
