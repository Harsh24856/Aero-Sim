import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AERO-SIM | Precision UAV Simulation",
  description: "Advanced telemetry and flight control systems for the next generation of aerospace engineering.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="app-shell text-on-surface min-h-screen flex flex-col overflow-x-hidden font-body-md selection:bg-tertiary selection:text-black">
        {children}
      </body>
    </html>
  );
}
