import ReifierProofs.Proofs.Grid
import ReifierProofs.Proofs.Trace
import ReifierProofs.Proofs.Decomp
import ReifierProofs.Proofs.Compiled
import ReifierProofs.Proofs.Norm
import ReifierProofs.Proofs.Gelu
import ReifierProofs.Proofs.Round
import ReifierProofs.Proofs.LayerReal
import ReifierProofs.Proofs.LayerBF16
import ReifierProofs.Proofs.AttnReal
import ReifierProofs.Proofs.AttnBF16
import ReifierProofs.Proofs.Logits
import ReifierProofs.Proofs.Assembly
import ReifierProofs.Proofs.LayerGain
import ReifierProofs.Proofs.LogitsE
import ReifierProofs.Proofs.AssemblyDeep

/-!
# The theorems, proved
-/

namespace Reifier

/-- **For every well-formed program on bits, the transformer compiled from it, in real
arithmetic, holds each output bit within `10^-9` and gives the right next token.** -/
theorem main_real : MainReal :=
  mainReal_of stmt_grid stmt_trace
    (layer_real_of stmt_decomp stmt_unit_dots stmt_compiled stmt_trace stmt_norm_close stmt_gelu)
    (attn_real_of stmt_grid stmt_trace) (logit_real_of stmt_grid stmt_norm_close)

/-- **For every well-formed program on bits whose units bf16 holds, the transformer compiled
from it, in bf16, holds each output bit exactly and gives the right next token.** -/
theorem main_bf16 : MainBF16 :=
  mainBF16_of stmt_grid stmt_trace stmt_round
    (layer_bf16_of stmt_decomp stmt_unit_dots stmt_compiled stmt_trace stmt_norm_exact stmt_round)
    (attn_bf16_of stmt_grid stmt_trace stmt_norm_exact stmt_round)
    (logit_bf16_of stmt_grid stmt_norm_exact stmt_round)

/-- **For a well-formed program on bits whose GLUs are not flat, the transformer compiled from it,
in real arithmetic, holds each output bit and gives the right next token, to a bound on depth.** -/
theorem main_real_deep : MainRealDeep :=
  mainRealDeep_of stmt_grid stmt_trace
    (layer_gain_of stmt_decomp stmt_unit_dots stmt_compiled stmt_trace stmt_norm_close stmt_gelu)
    (attn_real_of stmt_grid stmt_trace) (logit_real_E_of stmt_grid stmt_norm_close)

/-- The bf16 theorem is about something: there is a rounding to bf16. -/
theorem rounding_exists : Stmt_rounding_exists := stmt_rounding_exists

end Reifier
