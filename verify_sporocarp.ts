#!/usr/bin/env tsx
// ============================================================================
// SPOROCARP — Independent Biophysical Verification Suite Runner
// Owned exclusively by Worker 4 (biophysics-auditor). Locked by ORACLE.sha256.
//
// Usage:
//   npx tsx verify_sporocarp.ts           # Verify ORACLE.sha256 + run Tiers 1, 2, 3
//   npx tsx verify_sporocarp.ts --tier12  # Verify ORACLE.sha256 + run Tiers 1 & 2
//   npx tsx verify_sporocarp.ts --tier3   # Verify ORACLE.sha256 + run Tier 3 (Playwright)
//   npx tsx verify_sporocarp.ts --only=T1.3,T2.1   # Debug subset of checks (Tier 3 id: T3.1)
//   npx tsx verify_sporocarp.ts --lock    # Generate ORACLE.sha256 (Worker 4 setup only)
// ============================================================================

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import type { OracleTestResult } from './tests/tier1_oracles.ts';

const ROOT_DIR = process.cwd();
const ORACLE_LOCK_FILE = path.join(ROOT_DIR, 'ORACLE.sha256');
const LOCKED_FILES = [
  'verify_sporocarp.ts',
  'tests/tier1_oracles.ts',
  'tests/tier2_coupling.ts',
  'tests/tier3_browser.ts',
] as const;

function sha256File(relPath: string): string {
  const abs = path.join(ROOT_DIR, relPath);
  const data = fs.readFileSync(abs);
  return crypto.createHash('sha256').update(data).digest('hex');
}

function writeOracleLock(): void {
  const lines = LOCKED_FILES.map((rel) => `${sha256File(rel)}  ${rel}`);
  fs.writeFileSync(ORACLE_LOCK_FILE, `${lines.join('\n')}\n`, 'utf8');
  console.log(`Locked ${LOCKED_FILES.length} oracle files into ORACLE.sha256:`);
  for (const line of lines) {
    console.log(`  ${line}`);
  }
}

function verifyOracleLock(): { ok: boolean; errors: string[] } {
  const errors: string[] = [];
  if (!fs.existsSync(ORACLE_LOCK_FILE)) {
    return { ok: false, errors: ['Missing ORACLE.sha256 integrity lock file!'] };
  }

  const raw = fs.readFileSync(ORACLE_LOCK_FILE, 'utf8').trim();
  const expectedMap = new Map<string, string>();
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const match = trimmed.match(/^([a-f0-9]{64})\s+(.+)$/i);
    if (!match) {
      errors.push(`Malformed line in ORACLE.sha256: ${trimmed}`);
      continue;
    }
    expectedMap.set(match[2].trim(), match[1].toLowerCase());
  }

  for (const rel of LOCKED_FILES) {
    const expected = expectedMap.get(rel);
    if (!expected) {
      errors.push(`Locked file missing from ORACLE.sha256: ${rel}`);
      continue;
    }
    const actual = sha256File(rel);
    if (actual !== expected) {
      errors.push(
        `SHA-256 MISMATCH for ${rel}! expected=${expected.slice(0, 16)}... actual=${actual.slice(0, 16)}...`,
      );
    }
  }

  return { ok: errors.length === 0, errors };
}

function formatMetricValue(v: number | string | boolean): string {
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'string') return v;
  if (!Number.isFinite(v)) return String(v);
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs < 1e-3 || abs >= 1e5) return v.toExponential(4);
  return Number(v.toFixed(6)).toString();
}

function printResult(res: OracleTestResult, elapsedMs: number): void {
  const badge = res.passed ? '[PASS]' : '[FAIL]';
  console.log(
    `\n${badge} ${res.id} (Tier ${res.tier}) — ${res.name} (${elapsedMs.toFixed(0)} ms)`,
  );
  for (const d of res.details) {
    console.log(`       ${d}`);
  }
  const metricEntries = Object.entries(res.metrics);
  if (metricEntries.length > 0) {
    console.log('       Metrics:');
    for (const [k, v] of metricEntries) {
      console.log(`         - ${k}: ${formatMetricValue(v)}`);
    }
  }
  if (res.error) {
    console.log(`       ERROR: ${res.error}`);
  }
}

async function main(): Promise<void> {
  const args = new Set(process.argv.slice(2));

  if (args.has('--lock')) {
    writeOracleLock();
    return;
  }

  const onlyArg = process.argv.slice(2).find((a) => a.startsWith('--only='));
  const only = onlyArg ? new Set(onlyArg.slice('--only='.length).split(',')) : null;
  const wanted = (id: string): boolean => only === null || only.has(id);

  const runTier12 = args.has('--tier12') || (!args.has('--tier12') && !args.has('--tier3'));
  const runTier3 = args.has('--tier3') || (!args.has('--tier12') && !args.has('--tier3'));

  console.log('============================================================================');
  console.log('SPOROCARP — Independent Biophysical Verification Suite');
  console.log('============================================================================');

  const lockCheck = verifyOracleLock();
  if (!lockCheck.ok) {
    console.error('\n[FAIL] ORACLE.sha256 INTEGRITY CHECK FAILED:');
    for (const e of lockCheck.errors) {
      console.error(`  - ${e}`);
    }
    process.exit(1);
  }
  console.log(`[PASS] ORACLE.sha256 verified (${LOCKED_FILES.length} locked files intact)`);

  const allResults: OracleTestResult[] = [];

  if (runTier12) {
    console.log('\n--- Running Tier 1: Independent Exact Analytical Physics Oracles ---');
    const t1 = await import('./tests/tier1_oracles.ts');
    const tier1Fns: Array<[string, () => OracleTestResult]> = [
      ['T1.1', t1.runT1_1_SoilHydrology],
      ['T1.2', t1.runT1_2_LockhartGrowth],
      ['T1.3', t1.runT1_3_ThinShellBuckling],
      ['T1.4', t1.runT1_4_BullerCatapult],
    ];
    for (const [id, fn] of tier1Fns) {
      if (!wanted(id)) continue;
      const start = performance.now();
      const res = fn();
      const elapsed = performance.now() - start;
      allResults.push(res);
      printResult(res, elapsed);
    }

    console.log('\n--- Running Tier 2: Emergent Coupling & Falsification Tests ---');
    const t2 = await import('./tests/tier2_coupling.ts');
    const tier2Fns: Array<[string, () => OracleTestResult]> = [
      ['T2.1', t2.runT2_1_DroughtHypoxiaWilt],
      ['T2.2', t2.runT2_2_Gravitropism],
      ['T2.3', t2.runT2_3_Co2Etiolation],
      ['T2.4', t2.runT2_4_FairyRing],
      ['T2.5', t2.runT2_5_DressaireEvaporativeLift],
    ];
    for (const [id, fn] of tier2Fns) {
      if (!wanted(id)) continue;
      const start = performance.now();
      const res = fn();
      const elapsed = performance.now() - start;
      allResults.push(res);
      printResult(res, elapsed);
    }
  }

  if (runTier3 && wanted('T3.1')) {
    console.log('\n--- Running Tier 3: Live Browser Visual & Interactive Audit ---');
    const t3 = await import('./tests/tier3_browser.ts');
    const start = performance.now();
    const res = await t3.runTier3BrowserAudit();
    const elapsed = performance.now() - start;
    allResults.push(res);
    printResult(res, elapsed);
  }

  const passedCount = allResults.filter((r) => r.passed).length;
  const totalCount = allResults.length;

  console.log('\n============================================================================');
  console.log(`SUMMARY: ${passedCount}/${totalCount} verification checks PASSED`);
  console.log('============================================================================');

  if (passedCount !== totalCount) {
    const failedIds = allResults.filter((r) => !r.passed).map((r) => r.id);
    console.error(`FAILED CHECKS: ${failedIds.join(', ')}`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Unhandled verifier crash:', err);
  process.exit(1);
});
