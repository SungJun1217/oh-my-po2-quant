export const meta = {
  name: 'po2-research-cycle',
  description: 'Phase 1 zoo cycle: subset + baseline sweep → design → plan → implement/review → dev screening → dev gate → failure analysis, then one held-out gate',
  whenToUse: 'Develop a model-agnostic Po2 PTQ method across benchmarks/phase1.yaml on feature/<topic>. args: {topic, goal, stopAfter?, maxCycles?, maxFixRounds?, evalConcurrency?, baseline?, openPr?}',
  phases: [
    { title: 'Setup', detail: 'create feature/<topic> from develop, read manifest, build/validate ImageNet screening subset' },
    { title: 'Baseline', detail: 'po2-perf-evaluator per dev model (screening + gate), proxy check, failure analysis' },
    { title: 'Design', detail: 'po2-algorithm-designer (opus)' },
    { title: 'Plan', detail: 'po2-architect (opus)' },
    { title: 'Implement', detail: 'po2-implementer (sonnet) ↔ po2-reviewer (opus) per step' },
    { title: 'Evaluate', detail: 'po2-perf-evaluator per dev model: screening, then full-val gate' },
    { title: 'Analyze', detail: 'po2-failure-analyst (opus)' },
    { title: 'Held-out', detail: 'po2-perf-evaluator per held-out model, run once' },
    { title: 'Finish', detail: 'push feature branch and open PR to develop (only if openPr)' },
  ],
}

// ---- args ----
const A = args || {}
if (!A.topic || !A.goal) throw new Error('args.topic and args.goal are required')
const TOPIC = A.topic
const GOAL = A.goal
const STOP_AFTER = A.stopAfter || 'all'        // 'baseline' | 'design' | 'plan' | 'implement' | 'dev' | 'all'
const MAX_CYCLES = A.maxCycles || 5
const MAX_FIX = A.maxFixRounds || 3
const CONC = A.evalConcurrency || 1            // parallel model evaluations; >1 only with separate GPUs
const RUN_BASELINE = A.baseline !== false
const OPEN_PR = A.openPr === true              // push + PR feature → develop at the end; off by default
const BRANCH = `feature/${TOPIC}`
const TRAILER = 'Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>'
const MANIFEST = 'benchmarks/phase1.yaml'

const DESIGN = `docs/design/${TOPIC}.md`
const PLAN = `docs/plans/${TOPIC}.md`

// ---- schemas ----
const MANIFEST_S = {
  type: 'object',
  properties: {
    status: { type: 'string' },
    dev: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, task: { type: 'string' } }, required: ['id', 'task'] } },
    heldout: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, task: { type: 'string' } }, required: ['id', 'task'] } },
    heldoutPassRate: { type: 'number' },
  },
  required: ['status', 'dev', 'heldout', 'heldoutPassRate'],
}
const DESIGN_S = {
  type: 'object',
  properties: {
    designPath: { type: 'string' },
    hwCompatible: { type: 'boolean', description: 'false if the method needs any feature the HW profile does not allow' },
    summary: { type: 'string' },
    openQuestions: { type: 'array', items: { type: 'string' } },
  },
  required: ['designPath', 'hwCompatible', 'summary'],
}
const PLAN_S = {
  type: 'object',
  properties: {
    planPath: { type: 'string' },
    designOk: { type: 'boolean', description: 'false if the design is mathematically wrong or not HW-compatible' },
    designIssues: { type: 'array', items: { type: 'string' } },
    hwBlockers: { type: 'array', items: { type: 'string' }, description: 'op semantics missing from the HW profile' },
    steps: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, title: { type: 'string' } }, required: ['id', 'title'] } },
  },
  required: ['planPath', 'designOk', 'steps'],
}
const IMPL_S = {
  type: 'object',
  properties: {
    filesChanged: { type: 'array', items: { type: 'string' } },
    testsPassed: { type: 'boolean' },
    testOutputTail: { type: 'string' },
    deviations: { type: 'array', items: { type: 'string' } },
    blocked: { type: 'boolean', description: 'true if the plan is ambiguous/contradictory and you stopped' },
    blockedReason: { type: 'string' },
  },
  required: ['filesChanged', 'testsPassed', 'blocked'],
}
const REVIEW_S = {
  type: 'object',
  properties: {
    pass: { type: 'boolean', description: 'true if no blocker findings' },
    planProblem: { type: 'boolean', description: 'true if the blocker is in the plan, not the code' },
    blockers: { type: 'array', items: { type: 'object', properties: { location: { type: 'string' }, summary: { type: 'string' }, failure: { type: 'string' } }, required: ['location', 'summary'] } },
  },
  required: ['pass', 'planProblem', 'blockers'],
}
const EVAL_S = {
  type: 'object',
  properties: {
    model: { type: 'string' },
    reportPath: { type: 'string' },
    verdict: { type: 'string', enum: ['ACCEPT', 'REJECT', 'INCONCLUSIVE', 'INVALID'] },
    lossRelPct: { type: 'number' },
    ciUpperPct: { type: 'number' },
    route: { type: 'string', enum: ['none', 'designer', 'architect', 'implementer', 'evaluator'] },
    feedback: { type: 'string', description: 'bug description for architect/implementer, or extra runs needed' },
  },
  required: ['model', 'reportPath', 'verdict', 'route', 'feedback'],
}
const CHECK_S = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    path: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['ok', 'notes'],
}
const ANALYSIS_S = {
  type: 'object',
  properties: {
    analysisPath: { type: 'string' },
    topCategories: { type: 'array', items: { type: 'string' } },
    suspectedBugs: { type: 'array', items: { type: 'string' } },
  },
  required: ['analysisPath', 'topCategories'],
}

// ---- helpers ----
// Run fn over items with at most CONC concurrent agents (GPU contention breaks determinism).
async function mapLimited(items, fn) {
  const out = []
  for (let i = 0; i < items.length; i += CONC) {
    const chunk = items.slice(i, i + CONC)
    const res = await parallel(chunk.map((it, j) => () => fn(it, i + j)))
    out.push(...res)
  }
  return out
}

const GIT_S = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    branch: { type: 'string' },
    commit: { type: 'string', description: 'HEAD sha after the operation' },
    notes: { type: 'string' },
  },
  required: ['ok', 'notes'],
}

// ---- git ----
const gitSetup = () => agent(
  `Prepare git for topic ${TOPIC}. If branch ${BRANCH} exists, check it out. Otherwise require branch develop to exist, ` +
  `check out develop and create ${BRANCH} from it. Fail (ok=false) if the working tree has uncommitted changes to tracked files ` +
  `or if develop does not exist. Never push, never discard changes, never touch main.`,
  { phase: 'Setup', label: 'git:branch', schema: GIT_S, effort: 'low' })

// Commit whatever the previous stage left in the tree. No-op if clean.
const gitCommit = (msg, phaseName) => agent(
  `On branch ${BRANCH} (verify you are on it; fail if not), stage all changes with \`git add -A\` and commit with this message:\n\n` +
  `${msg}\n\n${TRAILER}\n\nIf there is nothing to commit, return ok=true with the current HEAD. Never push, amend or switch branches.`,
  { phase: phaseName, label: `git:commit`, schema: GIT_S, effort: 'low' })

const openPr = (summary) => agent(
  `Push ${BRANCH} to origin and open a pull request into develop with gh. Use merge-commit style (do not squash). ` +
  `Title: "${TOPIC}". Body: the summary below, then links to docs/design/${TOPIC}.md, docs/plans/${TOPIC}.md and docs/results/${TOPIC}/, ` +
  `ending with the line: 🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nSummary:\n${summary}`,
  { phase: 'Finish', label: 'git:pr', schema: GIT_S, effort: 'low' })

// ---- stages ----
const readManifest = () => agent(
  `Read ${MANIFEST} and return status, dev and heldout model lists (id, task), and criterion.heldout_pass_rate as heldoutPassRate.`,
  { phase: 'Setup', label: 'manifest', schema: MANIFEST_S, effort: 'low' })

const evalModel = (mode, m, feedback, tier) => agent(
  `Mode: ${mode}.${tier ? ` Tier: ${tier}.` : ''} Model: ${m.id} (task: ${m.task}) from ${MANIFEST}. Topic: ${TOPIC}` +
  (mode === 'baseline' ? '' : ` (design ${DESIGN}, plan ${PLAN})`) + '.\n' +
  `Judge this model only, with INT8 bias. Route: none if ACCEPT or REJECT on algorithm grounds; ` +
  `evaluator if only more runs are needed; architect/implementer for bugs, nondeterminism or missing logging.` +
  (feedback ? `\n\nPrevious attempt asked for:\n${feedback}` : ''),
  { agentType: 'po2-perf-evaluator', phase: mode === 'heldout' ? 'Held-out' : (mode === 'baseline' ? 'Baseline' : 'Evaluate'), label: `${mode}${tier ? `-${tier}` : ''}:${m.id}`, schema: EVAL_S })

const evalCheck = (mode, prompt) => agent(`Mode: ${mode}. ${prompt}`,
  { agentType: 'po2-perf-evaluator', phase: mode === 'subset' ? 'Setup' : 'Baseline', label: mode, schema: CHECK_S })

// Evaluate a list of models; re-run INCONCLUSIVE ones once.
async function evalZoo(mode, models, tier) {
  const res = await mapLimited(models, m => evalModel(mode, m, null, tier).then(async r => {
    if (r && r.route === 'evaluator') return (await evalModel(mode, m, r.feedback, tier)) || r
    return r
  }))
  return res.map((r, i) => r || { model: models[i].id, verdict: 'INVALID', route: 'architect', feedback: 'evaluator agent died', reportPath: '' })
}

const analyze = (source, cycle) => agent(
  `Build the failure taxonomy for ${source === 'baseline' ? 'the baseline sweep (docs/results/baseline/)' : `topic ${TOPIC}, cycle ${cycle} (docs/results/${TOPIC}/dev/, screening and gate tiers)`}. ` +
  `Write docs/analysis/${source === 'baseline' ? 'baseline' : TOPIC}-c${cycle}.md.`,
  { agentType: 'po2-failure-analyst', phase: 'Analyze', label: `analyze:c${cycle}`, schema: ANALYSIS_S })

const design = (feedback) => agent(
  `Topic: ${TOPIC}\nGoal: ${GOAL}\nWrite or revise the design doc at ${DESIGN}.` +
  (feedback ? `\n\nInput for this revision:\n${feedback}` : ''),
  { agentType: 'po2-algorithm-designer', phase: 'Design', label: 'design', schema: DESIGN_S })

const plan = (feedback) => agent(
  `Analyze ${DESIGN} and write or revise the implementation plan at ${PLAN}. Return the ordered list of small implementation steps.` +
  (feedback ? `\n\nAddress this feedback:\n${feedback}` : ''),
  { agentType: 'po2-architect', phase: 'Plan', label: 'plan', schema: PLAN_S })

const implement = (step, feedback) => agent(
  `Implement step ${step.id} ("${step.title}") of ${PLAN}, with its tests, and run the tests.` +
  (feedback ? `\n\nFix these review blockers first:\n${feedback}` : ''),
  { agentType: 'po2-implementer', phase: 'Implement', label: `impl:${step.id}`, schema: IMPL_S })

const review = (step) => agent(
  `Review the uncommitted diff for step ${step.id} ("${step.title}") of ${PLAN}. Run the tests.`,
  { agentType: 'po2-reviewer', phase: 'Implement', label: `review:${step.id}`, schema: REVIEW_S })

async function implementAll(steps) {
  for (const step of steps) {
    let feedback = null
    let passed = false
    for (let round = 1; round <= MAX_FIX; round++) {
      const impl = await implement(step, feedback)
      if (!impl) return { ok: false, to: 'stop', why: `implementer died on ${step.id}` }
      if (impl.blocked) return { ok: false, to: 'architect', why: `step ${step.id} blocked: ${impl.blockedReason}` }
      const rev = await review(step)
      if (!rev) return { ok: false, to: 'stop', why: `reviewer died on ${step.id}` }
      if (rev.pass && impl.testsPassed) {
        const c = await gitCommit(`feat(${TOPIC}): ${step.id} ${step.title}`, 'Implement')
        if (!c || !c.ok) return { ok: false, to: 'stop', why: `commit failed for ${step.id}: ${c ? c.notes : 'agent died'}` }
        passed = true; break
      }
      if (rev.planProblem) return { ok: false, to: 'architect', why: rev.blockers.map(b => `${b.location}: ${b.summary}`).join('\n') }
      feedback = rev.blockers.map(b => `- ${b.location}: ${b.summary}${b.failure ? ` (${b.failure})` : ''}`).join('\n') ||
        `Tests failing:\n${impl.testOutputTail || ''}`
      log(`${step.id}: round ${round}/${MAX_FIX} failed review`)
    }
    if (!passed) return { ok: false, to: 'stop', why: `${step.id} still failing after ${MAX_FIX} fix rounds` }
    log(`${step.id} passed review`)
  }
  return { ok: true }
}

const scoreboard = (res) => res.map(r => `${r.model}=${r.verdict}${r.lossRelPct != null ? `(${r.lossRelPct}%)` : ''}`).join(', ')

// ---- setup ----
phase('Setup')
const g = await gitSetup()
if (!g || !g.ok) return { status: 'stopped', why: `git setup failed: ${g ? g.notes : 'agent died'}` }
log(`on ${BRANCH}`)
const manifest = await readManifest()
if (!manifest) return { status: 'error', why: 'could not read manifest' }
if (manifest.status !== 'frozen') log(`WARNING: ${MANIFEST} status is "${manifest.status}", not "frozen". Results may not be comparable across runs.`)
const DEV = manifest.dev
const HELDOUT = manifest.heldout
const HELDOUT_RATE = manifest.heldoutPassRate || 0.9
log(`dev ${DEV.length} models, held-out ${HELDOUT.length} models, eval concurrency ${CONC}`)

// ImageNet screening subset: build once (FP32 outputs only), reuse if already frozen and valid.
const sub = await evalCheck('subset', `Build or verify the ImageNet screening subset defined in ${MANIFEST} (eval_tiers.screening.classification) using the dev classification models. Reuse it if it already exists and passes its checks.`)
if (!sub || !sub.ok) return { status: 'stopped', why: 'screening subset could not be built/validated', detail: sub }
log(`screening subset ok: ${sub.path || ''}`)
await gitCommit(`eval(${TOPIC}): screening subset`, 'Setup')

const history = []
let feedback = null

// ---- baseline sweep ----
if (RUN_BASELINE) {
  phase('Baseline')
  const base = await evalZoo('baseline', DEV)
  history.push({ stage: 'baseline', scoreboard: scoreboard(base) })
  log(`baseline: ${scoreboard(base)}`)
  const bugs = base.filter(r => r.verdict === 'INVALID')
  if (bugs.length) return { status: 'stopped', why: 'baseline sweep invalid — fix infra first', invalid: bugs, history }
  const proxy = await evalCheck('proxy-check', `Using all baseline reports in docs/results/baseline/, check that screening tracks gate per ${MANIFEST} eval_tiers.proxy_check.`)
  history.push({ stage: 'proxy-check', ...(proxy || {}) })
  if (!proxy || !proxy.ok) return { status: 'stopped', why: 'screening does not track gate — rebuild the subset before iterating', detail: proxy, history }
  const an = await analyze('baseline', 0)
  await gitCommit(`eval(${TOPIC}): baseline sweep`, 'Baseline')
  if (an) { history.push({ stage: 'baseline-analysis', ...an }); feedback = `Baseline failure taxonomy: ${an.analysisPath}\nTop categories: ${an.topCategories.join('; ')}` }
  if (STOP_AFTER === 'baseline') return { status: 'paused', after: 'baseline', history }
}

// ---- main loop ----
let next = 'designer'
let steps = null

for (let cycle = 1; cycle <= MAX_CYCLES; cycle++) {
  log(`cycle ${cycle}/${MAX_CYCLES}: start at ${next}`)

  if (next === 'designer') {
    phase('Design')
    const d = await design(feedback)
    if (!d) return { status: 'error', why: 'designer died', history }
    history.push({ cycle, stage: 'design', ...d })
    await gitCommit(`design(${TOPIC}): cycle ${cycle}`, 'Design')
    if (!d.hwCompatible) return { status: 'stopped', why: 'design is not HW-compatible', design: d, history }
    if (STOP_AFTER === 'design') return { status: 'paused', after: 'design', design: d, history }
    next = 'architect'; feedback = null
  }

  if (next === 'architect') {
    phase('Plan')
    let p = await plan(feedback)
    if (!p) return { status: 'error', why: 'architect died', history }
    if (!p.designOk) {
      const d = await design(`Architect found design problems:\n${(p.designIssues || []).join('\n')}`)
      if (!d || !d.hwCompatible) return { status: 'stopped', why: 'design rejected by architect and not fixed', history }
      p = await plan(null)
      if (!p || !p.designOk) return { status: 'stopped', why: 'architect still rejects design', plan: p, history }
    }
    history.push({ cycle, stage: 'plan', planPath: p.planPath, steps: p.steps.length })
    await gitCommit(`plan(${TOPIC}): cycle ${cycle}`, 'Plan')
    if (p.hwBlockers && p.hwBlockers.length) return { status: 'needs_hw_info', hwBlockers: p.hwBlockers, plan: p, history }
    if (STOP_AFTER === 'plan') return { status: 'paused', after: 'plan', plan: p, history }
    steps = p.steps
    next = 'implementer'; feedback = null
  }

  if (next === 'implementer') {
    phase('Implement')
    if (!steps) steps = [{ id: `fix-c${cycle}`, title: `Fix issues from evaluation: ${feedback}` }]
    const r = await implementAll(steps)
    steps = null
    history.push({ cycle, stage: 'implement', ...r })
    if (!r.ok && r.to === 'architect') { next = 'architect'; feedback = r.why; continue }
    if (!r.ok) return { status: 'stopped', why: r.why, history }
    if (STOP_AFTER === 'implement') return { status: 'paused', after: 'implement', history }
    next = 'evaluator'; feedback = null
  }

  if (next === 'evaluator') {
    phase('Evaluate')
    let tier = 'screening'
    let res = await evalZoo('dev', DEV, 'screening')
    history.push({ cycle, stage: 'dev-screening', scoreboard: scoreboard(res) })
    await gitCommit(`eval(${TOPIC}): dev screening cycle ${cycle}`, 'Evaluate')
    log(`dev screening c${cycle}: ${scoreboard(res)}`)
    const screenBugs = res.some(r => r.verdict === 'INVALID' || r.route === 'architect' || r.route === 'implementer')
    if (!screenBugs && res.every(r => r.verdict === 'ACCEPT')) {
      // every model passes screening → confirm on the full validation sets
      tier = 'gate'
      res = await evalZoo('dev', DEV, 'gate')
      history.push({ cycle, stage: 'dev-gate', scoreboard: scoreboard(res) })
      await gitCommit(`eval(${TOPIC}): dev gate cycle ${cycle}`, 'Evaluate')
      log(`dev gate c${cycle}: ${scoreboard(res)}`)
    }

    // bugs first: an invalid run says nothing about the algorithm
    const bugs = res.filter(r => r.verdict === 'INVALID' || r.route === 'architect' || r.route === 'implementer')
    if (bugs.length) {
      next = bugs.some(r => r.route === 'architect') ? 'architect' : 'implementer'
      feedback = bugs.map(r => `- ${r.model}: ${r.feedback}`).join('\n')
      continue
    }

    const allPass = tier === 'gate' && res.every(r => r.verdict === 'ACCEPT')
    if (!allPass) {
      if (STOP_AFTER === 'dev') return { status: 'paused', after: 'dev', devResults: res, history }
      phase('Analyze')
      const an = await analyze('dev', cycle)
      if (!an) return { status: 'error', why: 'failure analyst died', history }
      history.push({ cycle, stage: 'analysis', ...an })
      await gitCommit(`analysis(${TOPIC}): cycle ${cycle}`, 'Analyze')
      if (an.suspectedBugs && an.suspectedBugs.length) {
        next = 'architect'; feedback = `Failure analyst suspects bugs/HW gaps:\n${an.suspectedBugs.join('\n')}`
        continue
      }
      const failing = res.filter(r => r.verdict !== 'ACCEPT').map(r => r.model)
      next = 'designer'
      feedback = `Dev models not yet within 1% (${tier} tier): ${failing.join(', ')}\nFailure taxonomy: ${an.analysisPath}\nTop categories: ${an.topCategories.join('; ')}`
      continue
    }

    if (STOP_AFTER === 'dev') return { status: 'dev_passed', devResults: res, history }
    const finish = async (status, extra) => {
      let pr = null
      if (OPEN_PR) { phase('Finish'); pr = await openPr(`${status}: ${JSON.stringify(extra)}`) }
      return { status, branch: BRANCH, pr, ...extra, history }
    }

    // ---- held-out gate: run once; details never flow back to design ----
    phase('Held-out')
    log('all dev models ACCEPT at gate — running held-out gate (once)')
    const ho = await evalZoo('heldout', HELDOUT, 'gate')
    const invalid = ho.filter(r => r.verdict === 'INVALID')
    const passed = ho.filter(r => r.verdict === 'ACCEPT').length
    const rate = passed / HELDOUT.length
    history.push({ cycle, stage: 'heldout', passed, total: HELDOUT.length, invalid: invalid.length })
    await gitCommit(`eval(${TOPIC}): held-out gate`, 'Held-out')
    if (invalid.length) return { status: 'heldout_invalid', why: 'held-out runs invalid — fix infra, then rerun the gate', invalid, history }
    if (rate >= HELDOUT_RATE) return await finish('accepted', { dev: scoreboard(res), heldout: { passed, total: HELDOUT.length, rate } })
    // Deliberately stop instead of looping: feeding held-out failures back would turn held-out into dev.
    return { status: 'heldout_failed', heldout: { passed, total: HELDOUT.length, rate, required: HELDOUT_RATE },
      note: 'Human decision needed: accept, or refresh the held-out set before further design iterations.', history }
  }
}

return { status: 'cycle_limit', next, feedback, history }
