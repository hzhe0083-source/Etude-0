# Learned effect interfaces

`src/etude/models.py` contains trainable PyTorch modules, not trained robot skills.
The CPU checks verify indexing, gradient and information-flow contracts; they do
not establish grasping, migration or closed-loop success.

## Requirements and tokens

`RequirementCodec(entity_dim, geometry_dim, relation_dim, event_dim, roles,
token_dim=64, interface="full", max_precedence_edges=4)` is shared by current and
remaining requirements. The expanded heads require new checkpoints; old heads
cannot be silently loaded as equivalent models.

- `encode(requirement, entity_features)` returns `[B,T*R,D]` tokens, time-major.
- `decode(tokens, entity_features, step_offsets, entity_ids)` returns a
  `DecodedRequirement`: binding categorical logits `[B,R,N+3]`, geometry
  `[B,T,R,Dg]`, independent relationship/event logits `[B,T,R,R,C/E]`, and
  same-shaped requirement-mask logits. Additional outputs are nonnegative
  geometry tolerances, continuous event-window endpoints, four edge-presence
  logits, and source/target endpoint logits over flattened event nodes.
- `forward(requirement, entity_features)` combines encode and decode.
- `decoded_requirement_loss(decoded, requirement, interface="full",
  evidence_valid=None)` supervises
  available values and requirement masks. A predicted mask cannot switch off a
  label. Invalid numeric labels are replaced before arithmetic, including NaNs.
  Optional per-view evidence has the same keys and shapes as `label_valid`.
  Non-unused roles unidentifiable from that view are supervised as `UNCERTAIN`,
  not forced to match the unobscured ground-truth object. Other semantic values
  and masks receive only available-view supervision.
- `DecodedRequirement.probabilities()` returns binding categorical probabilities
  and independent relationship/event Bernoulli probabilities. For a two-category
  comparison, convert each Bernoulli `p` to `[p,1-p]`; do not softmax across
  different relationships.
- `materialize(threshold=0.5, interface="full", min_binding_confidence=0.5,
  min_binding_margin=0.1)` builds a runtime requirement. Its available-field
  flags mean that outputs exist, not that predictions are true annotations.
  Never write this materialization back as ground-truth supervision. Active empty
  requirements still receive an infinite cost.

The final binding columns are `N=UNUSED`, `N+1=UNMATCHED`, `N+2=UNCERTAIN`,
materialized as `-1`, `-2`, `-3`. G uses separate learned status embeddings,
never a fabricated entity at slot zero. `UNUSED` cannot participate in a required
condition. `UNMATCHED` and `UNCERTAIN` retain requirements but block execution.

A binding passes only when the top category clears both probability and
top-two margin thresholds. Uniform scores never count as evidence for entity
zero. Defaults are conservative diagnostic settings, not calibrated confidence
claims; formal runs must lock thresholds on validation data. If `UNUSED`
conflicts with a required role, materialization returns `UNCERTAIN` instead
of deleting the role's requirements.

The codec binds roles to observed entity features, not to numeric entity IDs.
Permuting the entity table and inverse-mapping bindings preserves encoded tokens;
binding output probabilities follow the entity permutation; the three status
columns remain fixed. Current and remaining
may have different time grids. The same role vocabulary, feature dimensions and
trained codec apply to both.

The `geometry` control keeps exactly the same token width, count and parameter
shapes as the full interface. Its encoder zeros relationship/event values,
requirement masks and event windows, omits ordering messages, and its decoder
loss omits those fields. Geometry tolerance remains part of the geometric
interface. Label validity is never a goal-token input. The main WAM
may still receive the same independent interaction auxiliary supervision in both
experiments; the geometry representation is not claimed to be information-pure.

## Allowed geometry and event timing

`geometry_tolerance` is a nonnegative half-width for every geometry coordinate:
the permitted interval is `[geometry-tolerance, geometry+tolerance]`. It has
separate data validity; G does not invent an unknown required bound.

`event_windows[B,T,R,R,E,2]` stores inclusive integer action-step intervals.
Event slots name requirements, not unique occurrence times. Positive events may
occur anywhere in their windows; negative events are forbidden throughout their
windows. Q predicts a positive lower endpoint and nonnegative width, rounded
to valid integer intervals at materialization.

`event_precedence[B,P,2]` stores directed edges over flattened `(T,R,R,E)` nodes.
`(-1,-1)` is an explicitly unused edge slot; validity distinguishes unused from
unknown. The model supports four slots and rejects larger inputs. G projects
concatenated source/destination descriptors, preserving direction. Q predicts
presence plus both endpoints. Known absent edges supervise presence; endpoint
loss only applies to known present edges.

Supervision canonicalizes edges after masking invisible endpoints: known edges
first, known empty slots next, unknown slots last. Changing an invisible value
cannot move a visible edge to another loss slot. Self-edges, duplicate edges,
cycles, non-required endpoints or contradictory windows at materialization are
explicit errors to report as rejected goals, never silently deleted constraints.

## Demonstration reader

`EffectReader(demo_dim, entity_dim, proprio_dim, embodiment_dim, roles,
token_dim=64)` consumes already-encoded demonstration tokens; it adds no image
encoder. Its forward arguments are:

```python
goals = reader(
    demo_tokens,       # B,S,Ddemo
    entity_history,    # B,L,N,Dentity, before task conditioning
    proprio_history,  # B,L,Dproprio
    embodiment,       # B,Dembodiment
    current_offsets,  # Tcurrent
    remaining_offsets,# Tremaining
    entity_present=entity_ids >= 0,  # required when padded slots are used
)
```

`goals.current` and `goals.remaining` are `[B,T*R,D]`. The reader attends to
demonstration tokens and observed scene entities. Entity order is a set axis;
observation history and time queries remain ordered. Demonstration tokens must
follow a fixed frame-major temporal/spatial order. The reader adds deterministic
sinusoidal token positions so reversing event order cannot become the same set
of inputs; upstream local motion features do not replace this global order.
Missing entities use the
explicit presence mask rather than becoming a false contact absence.

## Physical predictor and auxiliary head

`CausalEffectPredictor(observation_dim, proprio_dim, action_dim, embodiment_dim,
geometry_dim, relation_dim, event_dim, hidden_dim=64)` accepts only:

```python
prediction = predictor(history, proprio, actions, embodiment,
                       entity_present=entity_ids >= 0)
# history B,L,N,Dobs; proprio B,L,Dproprio; actions B,H,Daction
```

It returns `PhysicalPrediction(geometry, relation_logits, event_logits,
prediction_uncertainty)`. Shapes are `[B,H,N,Dg]`, `[B,H,N,N,C]`,
`[B,H,N,N,E]`, and `[B,H,N]`. Geometry is relative to the present observed state;
relations are states at each step, and events describe intervals ending at that
step. A shared per-entity recurrent model and ordered pair readouts preserve
entity permutation. The action recurrence is causal: future actions cannot
change a prefix prediction. There is no goal, demonstration, text or task-cache
argument. The caller must supply features from before task conditioning.

`physical_prediction_loss(prediction, outcome)` selects one-based action steps
using `outcome.step_offsets`. Data validity alone gates supervised residuals.
The uncertainty head learns detached, per-entity residual magnitude, including
incident relationships. This is not a success probability or calibrated risk;
calibration and rejection thresholds require independent validation data.

`TemporalInteractionHead(feature_dim, relation_dim, event_dim, hidden_dim=64)`
is a separate training-only head. `head(phi, offsets)` returns relationship and
event logits `[B,K,N,N,C/E]`. `phi` is entity-aligned `[B,N,D]` or `[B,N,S,D]`,
where only `S` is pooled. Native video hidden states must first be pooled/aligned
with observation-derived entity masks. A raw video-patch axis is not an entity
axis. Neither the head nor alignment metadata may directly read goals or
demonstration tokens. This head is not the action-conditioned predictor.

## Matching costs and time

```python
scores = effect_cost(prediction, requirement, entity_visible, prefix_steps,
                     field_weights=None, uncertainty_weight=1.0, active=None,
                     event_threshold=0.5)
```

The scene entity table must have the same order as the predictor history.
`step_offsets` selects geometry and relation times from the complete candidate
window. Geometry error is zero within the allowed interval and squared distance
outside it. Relations use Brier error at their specified steps. Event Brier
error uses the largest occurrence score in the annotated window: positive
requirements reward occurrence, negatives penalize any occurrence. Errors are averaged
within each required field, then combined using fixed positive field weights.
The weights are calibration knobs, locked on validation data before evaluation.

The deployed prefix never moves a terminal requirement earlier. `prefix_steps`
only selects an extra early-window uncertainty contribution; the cost still
checks each requirement at its own time. Goals outside the candidate horizon
raise a clear error: callers must plan a longer window or provide an explicit
local-progress requirement, not silently omit the goal.

Precedence uses one common assignment of occurrence times with a locked
`event_threshold`. A topological traversal selects each event's earliest
in-window occurrence after all its predecessors. For strict-before DAGs this is
exact for feasibility: an earlier predecessor cannot harm a later choice.
Failure yields infinite cost. The checker cannot satisfy `A < B` with a late B
and `B < C` with a different early B. It does not plan robot actions, and its
thresholded predictions are not proof of real task success.

Uncertainty contributes a nonnegative additive term for involved entities;
increasing it cannot reduce a fixed requirement's cost. Missing required entity
observations, unresolved bindings, unknown required semantics, or active empty requirements yield
`inf` for rejection. Malformed contracts raise errors rather than being repaired.
An explicit inactive flag returns zero only after the caller independently
verified completion or a permitted idle state. Costs do not certify hardware
safety, and complete-candidate predictions are not claims about prefix-and-replan
closed-loop outcomes.

Run the interface checks with:

```bash
python -m unittest discover -s tests -p test_models.py -v
```
# Requirement annotation coverage

`label_valid` is loss/annotation metadata, not a semantic token input. G encodes
required values and requirement masks only. Executable teacher goals must have
known values for their required fields; encoding an unknown required value fails
explicitly rather than inventing a zero target. Unrequired fields may be missing,
and physical outcomes retain sparse labels for F and interaction supervision.
