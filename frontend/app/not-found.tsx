import Link from "next/link";
import Navbar from "@/components/Navbar";

// Next's built-in 404 set black text on this app's near-black body: the page looked empty.
export default function NotFound() {
  return (
    <>
      <Navbar />
      <main className="relative flex flex-1 items-center justify-center px-6 pt-16">
        <div className="absolute inset-0 grid-bg pointer-events-none opacity-50" />
        <div className="relative max-w-md text-center">
          <p className="font-mono text-[11px] font-bold uppercase tracking-[0.16em] text-tertiary">Error 404 // off the chart</p>
          <h1 className="mt-3 font-headline-display text-4xl font-bold uppercase tracking-tight text-primary">Page not found</h1>
          <p className="mt-3 text-sm leading-relaxed text-on-surface-variant">
            Nothing is flying at this address. It may have moved, or the link is mistyped.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <Link href="/home" className="rounded bg-tertiary px-5 py-3 font-mono text-[11px] font-bold uppercase tracking-wider text-black shadow-[2px_2px_0px_#000000] transition-transform hover:-translate-y-0.5">
              Back to home
            </Link>
            <Link href="/engine" className="rounded border border-outline-variant/60 px-5 py-3 font-mono text-[11px] font-bold uppercase tracking-wider text-primary transition-colors hover:border-tertiary hover:text-tertiary">
              Pick an engine
            </Link>
          </div>
        </div>
      </main>
    </>
  );
}
