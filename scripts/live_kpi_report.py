#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import text

from src.data.postgres import engine


def q_scalar(sql: str):
    with engine.connect() as c:
        return c.execute(text(sql)).scalar()


def q_rows(sql: str):
    with engine.connect() as c:
        return c.execute(text(sql)).fetchall()


def pct(a: float, b: float) -> float:
    return (100.0 * a / b) if b else 0.0


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"live_kpis_{ts}.md"

    total = q_scalar("""
        select count(*) from placed_bets
        where created_at > now() - interval '48 hours'
          and data_source in ('paper_signal','live_trade')
    """) or 0

    playable = q_scalar("""
        select count(*) from placed_bets
        where created_at > now() - interval '48 hours'
          and data_source in ('paper_signal','live_trade')
          and stake > 0
          and coalesce(meta_features->>'reject_reason','') = ''
    """) or 0

    at_cap = q_scalar("""
        select count(*) from placed_bets
        where created_at > now() - interval '48 hours'
          and data_source in ('paper_signal','live_trade')
          and stake >= 1.5
    """) or 0

    avg_ev = q_scalar("""
        select avg((meta_features->>'expected_value')::float)
        from placed_bets
        where created_at > now() - interval '48 hours'
          and data_source in ('paper_signal','live_trade')
          and meta_features ? 'expected_value'
    """) or 0.0

    high_ev = q_scalar("""
        select count(*) from placed_bets
        where created_at > now() - interval '48 hours'
          and data_source in ('paper_signal','live_trade')
          and coalesce((meta_features->>'expected_value')::float,0) > 0.15
    """) or 0

    by_reason = q_rows("""
        select coalesce(nullif(meta_features->>'reject_reason',''),'PLAYABLE') reason,
               count(*) n
        from placed_bets
        where created_at > now() - interval '48 hours'
          and data_source in ('paper_signal','live_trade')
        group by 1
        order by n desc
        limit 12
    """)

    by_sport = q_rows("""
        select sport, count(*) n,
               avg(stake) avg_stake
        from placed_bets
        where created_at > now() - interval '48 hours'
          and data_source in ('paper_signal','live_trade')
        group by 1
        order by n desc
        limit 12
    """)

    p50 = q_scalar("""
        select percentile_cont(0.5) within group (order by stake)
        from placed_bets
        where created_at > now() - interval '48 hours'
          and data_source in ('paper_signal','live_trade')
    """) or 0.0

    p90 = q_scalar("""
        select percentile_cont(0.9) within group (order by stake)
        from placed_bets
        where created_at > now() - interval '48 hours'
          and data_source in ('paper_signal','live_trade')
    """) or 0.0

    lines = []
    lines.append(f"# Live KPI Report ({ts})")
    lines.append("")
    lines.append("## Core")
    lines.append(f"- total_candidates_48h: **{total}**")
    lines.append(f"- playable_48h: **{playable}** ({pct(playable,total):.1f}%)")
    lines.append(f"- at_cap_48h: **{at_cap}** ({pct(at_cap,total):.1f}%)")
    lines.append(f"- avg_ev_48h: **{avg_ev:.4f}**")
    lines.append(f"- high_ev_gt_0.15_48h: **{high_ev}** ({pct(high_ev,total):.1f}%)")
    lines.append(f"- stake_p50_48h: **{p50:.2f}**")
    lines.append(f"- stake_p90_48h: **{p90:.2f}**")
    lines.append("")

    lines.append("## Reject Reasons (Top)")
    for reason, n in by_reason:
        lines.append(f"- {reason}: {n}")
    lines.append("")

    lines.append("## By Sport")
    for sport, n, avg_stake in by_sport:
        lines.append(f"- {sport}: n={n}, avg_stake={float(avg_stake or 0):.2f}")

    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
