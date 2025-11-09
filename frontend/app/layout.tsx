export const metadata = {
  title: "EX-AI-AR",
  description: "Explainable AI + AR viewer"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50">{children}</body>
    </html>
  );
}
