# Flujo de Aprobación de PRs — 2 Capas

Este documento describe el proceso obligatorio para fusionar cambios en `main`.
El sistema tiene **dos capas de aprobación** que funcionan en cascada:
ambas deben estar en verde para habilitar el botón de merge.

---

## Resumen visual del flujo

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. ISSUE se abre (explica el "por qué")                        │
│    └─> Compañero lo lee, comenta, y si tiene sentido           │
│        le pone la etiqueta  ✅  "approved" (verde)              │
│                                                                 │
│ 2. PR se abre con "Closes #N" en título/cuerpo                 │
│    └─> CAPA 1 (bot): Check automático verifica:                │
│         • ¿El PR menciona un issue?                            │
│         • ¿Ese issue tiene label "approved"?                   │
│         Si NO → ❌ check rojo, NO se puede mergear              │
│                                                                 │
│    └─> CAPA 2 (humano): Compañero revisa código vs issue       │
│         • Si todo bien → ✅ Approve en la UI de GitHub          │
│                                                                 │
│ 3. Solo con CAPA 1 verde + CAPA 2 verde → 🟢 Merge habilitado  │
└─────────────────────────────────────────────────────────────────┘
```

---

## CAPA 1 — Check automático (`pr-issue-check.yml`)

Se ejecuta en **cada PR** abierto/editado/sincronizado contra `main`.

### Qué valida
1. **El PR menciona al menos un issue** usando cualquiera de estos patrones:
   - `Closes #123`
   - `Fixes #123`
   - `Resolves #123`
   - `Relacionado a #123`
   - `Related to #123`
   - O simplemente `#123` en título/cuerpo (fallback)

2. **Ese issue existe** en el repositorio.

3. **Ese issue tiene la etiqueta `approved`** (verde, creada por un humano).

### Estados del check

| Situación | Resultado | Mensaje |
|-----------|-----------|---------|
| PR sin ningún `#N` | ❌ **FAIL** | `El PR no menciona ningún issue. Agrega "Closes #N"...` |
| Issue mencionado no existe | ❌ **FAIL** | `El issue #N no existe en este repositorio.` |
| Issue existe pero **sin** label `approved` | ❌ **FAIL** | `El issue #N NO tiene la etiqueta "approved". Pídele a tu compañero que la agregue...` |
| Issue existe **y tiene** label `approved` | ✅ **PASS** | `Todos los issues vinculados están aprobados.` |

### Cómo se ve en la UI de GitHub

**❌ Fallo por issue sin aprobar:**
```
❌ Verify linked issue is approved
   El issue #5 NO tiene la etiqueta "approved".
   Pídele a tu compañero que la agregue antes de abrir el PR.
```

**✅ Éxito:**
```
✅ Verify linked issue is approved
   ✅ Issue #5 tiene etiqueta "approved".
   ✅ Todos los issues vinculados están aprobados.
```

---

## CAPA 2 — Revisión humana (GitHub PR Review)

Una vez que la CAPA 1 está en verde, **un compañero** (no el autor) debe:

1. Abrir el PR en GitHub.
2. Leer el **issue vinculado** (clic en el enlace del check o buscar #N).
3. Revisar el **código del PR** y confirmar que **resuelve lo que dice el issue**.
4. Clic en **Files changed** → **Review changes** → **Approve**.
5. Opcional: dejar comentarios si hay dudas.

> **Regla de oro**: El revisor NO aprueba el PR si el código no resuelve el issue aprobado, aunque la CAPA 1 esté en verde. La CAPA 2 es el filtro semántico.

---

## Paso a paso para el autor del cambio

### 1. Abrir el Issue (antes de codear)
- Ve a **Issues → New issue**.
- Título claro: `Fix: memory leak in mapper` / `Feature: add CSV export`.
- Cuerpo: explica el **problema** y la **solución esperada** (mini-PRD).
- **No** pongas la etiqueta `approved` tú mismo — la pone el compañero.

### 2. Esperar la etiqueta `approved`
- Tu compañero lee, comenta, pregunta si hace falta.
- Si está de acuerdo, **él** clica en **Labels → approved** (verde).
- El issue queda con la etiqueta verde ✅.

### 3. Codear y abrir el PR
- Crea tu rama: `git checkout -b fix/memory-leak`.
- Haz tus commits (usa conventional commits: `fix(mapper): ...`).
- Push y abre el PR en GitHub.
- **En el título o cuerpo del PR, OBLIGATORIO**:
  ```
  Closes #123
  ```
  (o `Fixes #123`, `Resolves #123`, `Relacionado a #123`).
- El check `Verify linked issue is approved` correrá solo.

### 4. Si el check falla
- **Sin issue mencionado**: agrega `Closes #N` al título/cuerpo y haz push.
- **Issue sin approved**: avisa a tu compañero para que ponga la etiqueta.
- **Issue inexistente**: corrige el número.

### 5. Pedir revisión humana
- Cuando CAPA 1 esté ✅ verde, asigna el PR a tu compañero (Reviewers).
- Él revisa, comenta si hace falta, y da **Approve**.

### 6. Merge
- Con **CAPA 1 verde + CAPA 2 verde**, el botón **Merge** se habilita.
- Merge squash o rebase según convención del equipo.
- El issue vinculado se cierra automáticamente por `Closes #N`.

---

## Paso a paso para el revisor (compañero)

### Aprobar un Issue (antes del PR)
1. Ve al issue en GitHub.
2. Lee título y cuerpo. ¿Tiene sentido? ¿Está claro el alcance?
3. Si sí → **Labels → approved** (verde).
4. Si no → comenta preguntando/aclarando. No pongas `approved` hasta que esté claro.

### Revisar el PR (CAPA 2)
1. Abre el PR asignado.
2. Clic en el issue vinculado (aparece en la descripción o en la barra lateral).
3. Compara: **¿El código resuelve EXACTAMENTE lo que pide el issue?**
4. Revisa Files changed. Comenta si ves:
   - Código que no pertenece al issue (scope creep).
   - Falta de tests para lo nuevo.
   - Nomenclatura, tipos, estilo.
5. Si todo bien → **Review changes → Approve**.
6. Si hay dudas → **Request changes** o **Comment**.

---

## Preguntas frecuentes

**¿Puedo poner yo mismo la etiqueta `approved` en mi issue?**
No. La etiqueta la debe poner **otra persona** (el revisor). Es el "segundo par de ojos" antes de codear.

**¿Qué pasa si el PR menciona varios issues?**
El check valida **todos**. Si *cualquiera* no tiene `approved`, el check falla. Todos deben estar aprobados.

**¿Puedo usar `Relacionado a #N` en vez de `Closes #N`?**
Sí, el check detecta ambos. Pero usa `Closes #N` o `Fixes #N` cuando el PR **cierra** el issue (GitHub lo cerrará solo al mergear). Usa `Relacionado a #N` solo si el PR toca el tema pero no lo cierra del todo.

**¿El check corre en draft PRs?**
No, solo en PRs listos para review (opened, edited, synchronize, reopened). Los drafts no disparan el check.

**¿Qué pasa si edito el PR y saco el `Closes #N`?**
El check vuelve a correr en cada `synchronize` (push nuevo). Si quitas la referencia, el check fallará.

---

## Comandos útiles

```bash
# Ver issues con label approved
gh issue list --repo SantiC57/clearsky-lidar --label approved --state all

# Ver issues SIN label approved
gh issue list --repo SantiC57/clearsky-lidar --label "!approved" --state open

# Crear issue y pedir aprobación
gh issue create --repo SantiC57/clearsky-lidar --title "Fix: ..." --body "..."

# Aprobar issue (como revisor)
gh issue edit 123 --repo SantiC57/clearsky-lidar --add-label approved
```