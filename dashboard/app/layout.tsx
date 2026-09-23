import type {Metadata} from "next";
import "./globals.css";
import "./overrides.css";

export const metadata: Metadata = {
  title: "EUR/USD Paper Trader",
  description: "15-minute automated demo trading dashboard",
};

export default function Layout({children}: {children: React.ReactNode}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
