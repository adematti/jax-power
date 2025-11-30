# Vendored from JAX (git commit 7a3df3e)
# Original copyright: Google LLC
# License: Apache 2.0 (see below)
# Modifications: None

from __future__ import annotations

from functools import partial
import operator
from typing import cast, Any

import numpy as np

from jax._src import dtypes
from jax import numpy as jnp
from jax._src.numpy.ufuncs import isposinf, isneginf, sinc
from jax._src.api import jit, jvp, vmap
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy import einsum as jnp_einsum
from jax._src.numpy import vectorize as jnp_vectorize
from jax._src.numpy.util import promote_args_inexact, promote_dtypes_inexact
from jax._src.ops import special as ops_special
from jax._src.third_party.scipy.betaln import betaln as _betaln_impl
from jax._src.typing import Array, ArrayLike
from jax._src.nn.functions import softmax as nn_softmax
from jax._src.nn.functions import log_softmax as nn_log_softmax


@jit
def sici(x: ArrayLike) -> tuple[Array, Array]:
  r"""Sine and cosine integrals.

  JAX implementation of :obj:`scipy.special.sici`.

  .. math::

    \mathrm{Si}(x) = \int_0^x \frac{\sin t}{t} \, dt

  .. math::

    \mathrm{Ci}(x) = \gamma + \ln(x) + \int_0^x \frac{\cos t - 1}{t} \, dt

  where :math:`\gamma` is the Euler–Mascheroni constant.

  Args:
    x: array-like, real-valued input.

  Returns:
    A tuple of two arrays, each with the same shape as `x`:
      - The first array contains the sine integral values `Si(x)`.
      - The second array contains the cosine integral values `Ci(x)`.

  See also:
    - :func:`jax.numpy.sinc`
  """

  x, = promote_args_inexact("sici", x)

  if dtypes.issubdtype(x.dtype, np.complexfloating):
    raise ValueError(
      f"Argument `x` to sici must be real-valued. Got dtype {x.dtype}."
    )

  x_abs = jnp.abs(x)

  si_series, ci_series = _sici_series(x_abs)
  si_asymp,  ci_asymp  = _sici_asympt(x_abs)
  si_approx, ci_approx  = _sici_approx(x_abs)

  cond1 = x_abs <= 4
  cond2 = (x_abs > 4) & (x_abs <= 1e9)

  si = jnp.select([cond1, cond2], [si_series, si_asymp], si_approx)
  ci = jnp.select([cond1, cond2], [ci_series, ci_asymp], ci_approx)

  si = jnp.sign(x) * si
  ci = jnp.where(isneginf(x), np.nan, ci)

  return si, ci

def _sici_approx(x: Array):
  # sici approximation valid for x >= 1E9
  si = (np.pi / 2) - jnp.cos(x) / x
  ci = jnp.sin(x) / x

  si = jnp.where(isposinf(x), np.pi / 2, si)
  ci = jnp.where(isposinf(x), 0.0, ci)

  return si, ci

def _sici_series(x: Array):
  # sici series valid for x >= 0 and x <= 4
  def si_series(x):
    # Values come from Cephes Implementation used by Scipy https://github.com/jeremybarnes/cephes/blob/60f27df395b8322c2da22c83751a2366b82d50d1/misc/sici.c
    SN = np.array([-8.39167827910303881427E-11,
      4.62591714427012837309E-8,
      -9.75759303843632795789E-6,
      9.76945438170435310816E-4,
      -4.13470316229406538752E-2,
      1.00000000000000000302E0], dtype=x.dtype)
    SD = np.array([ 2.03269266195951942049E-12,
      1.27997891179943299903E-9,
      4.41827842801218905784E-7,
      9.96412122043875552487E-5,
      1.42085239326149893930E-2,
      9.99999999999999996984E-1], dtype=x.dtype)
    t = x * x
    return (x * jnp.polyval(SN, t)) / jnp.polyval(SD, t)

  def ci_series(x):
    # Values come from Cephes Implementation used by Scipy https://github.com/jeremybarnes/cephes/blob/60f27df395b8322c2da22c83751a2366b82d50d1/misc/sici.c
    CN = np.array([ 2.02524002389102268789E-11,
      -1.35249504915790756375E-8,
      3.59325051419993077021E-6,
      -4.74007206873407909465E-4,
      2.89159652607555242092E-2,
      -1.00000000000000000080E0], dtype=x.dtype)
    CD = np.array([ 4.07746040061880559506E-12,
      3.06780997581887812692E-9,
      1.23210355685883423679E-6,
      3.17442024775032769882E-4,
      5.10028056236446052392E-2,
      4.00000000000000000080E0], dtype=x.dtype)
    t = x * x
    return np.euler_gamma + jnp.log(x) + t * jnp.polyval(CN, t) / jnp.polyval(CD, t)

  si = jnp.where(
    x == 0,
    0.0,
    si_series(x)
  )

  ci = jnp.where(
    x == 0,
    -np.inf,
    ci_series(x)
  )

  return si, ci

def _sici_asympt(x: Array):
  # sici asympt valid for x > 4 & x <= 1E9
  s = jnp.sin(x)
  c = jnp.cos(x)
  z = 1.0 / (x * x)

  # Values come from Cephes Implementation used by Scipy https://github.com/jeremybarnes/cephes/blob/60f27df395b8322c2da22c83751a2366b82d50d1/misc/sici.c
  FN4 = np.array([
    4.23612862892216586994E0,
    5.45937717161812843388E0,
    1.62083287701538329132E0,
    1.67006611831323023771E-1,
    6.81020132472518137426E-3,
    1.08936580650328664411E-4,
    5.48900223421373614008E-7,
  ], dtype=x.dtype)
  FD4 = np.array([
    1,
    8.16496634205391016773E0,
    7.30828822505564552187E0,
    1.86792257950184183883E0,
    1.78792052963149907262E-1,
    7.01710668322789753610E-3,
    1.10034357153915731354E-4,
    5.48900252756255700982E-7,
  ], dtype=x.dtype)
  GN4 = np.array([
    8.71001698973114191777E-2,
    6.11379109952219284151E-1,
    3.97180296392337498885E-1,
    7.48527737628469092119E-2,
    5.38868681462177273157E-3,
    1.61999794598934024525E-4,
    1.97963874140963632189E-6,
    7.82579040744090311069E-9,
  ], dtype=x.dtype)
  GD4 = np.array([
    1,
    1.64402202413355338886E0,
    6.66296701268987968381E-1,
    9.88771761277688796203E-2,
    6.22396345441768420760E-3,
    1.73221081474177119497E-4,
    2.02659182086343991969E-6,
    7.82579218933534490868E-9,
  ], dtype=x.dtype)

  FN8 = np.array([
    4.55880873470465315206E-1,
    7.13715274100146711374E-1,
    1.60300158222319456320E-1,
    1.16064229408124407915E-2,
    3.49556442447859055605E-4,
    4.86215430826454749482E-6,
    3.20092790091004902806E-8,
    9.41779576128512936592E-11,
    9.70507110881952024631E-14,
  ], dtype=x.dtype)
  FD8 = np.array([
    1.0,
    9.17463611873684053703E-1,
    1.78685545332074536321E-1,
    1.22253594771971293032E-2,
    3.58696481881851580297E-4,
    4.92435064317881464393E-6,
    3.21956939101046018377E-8,
    9.43720590350276732376E-11,
    9.70507110881952025725E-14,
  ], dtype=x.dtype)
  GN8 = np.array([
    6.97359953443276214934E-1,
    3.30410979305632063225E-1,
    3.84878767649974295920E-2,
    1.71718239052347903558E-3,
    3.48941165502279436777E-5,
    3.47131167084116673800E-7,
    1.70404452782044526189E-9,
    3.85945925430276600453E-12,
    3.14040098946363334640E-15,
  ], dtype=x.dtype)
  GD8 = np.array([
    1.0,
    1.68548898811011640017E0,
    4.87852258695304967486E-1,
    4.67913194259625806320E-2,
    1.90284426674399523638E-3,
    3.68475504442561108162E-5,
    3.57043223443740838771E-7,
    1.72693748966316146736E-9,
    3.87830166023954706752E-12,
    3.14040098946363335242E-15,
  ], dtype=x.dtype)

  f4 = jnp.polyval(FN4, z) / (x * jnp.polyval(FD4, z))
  g4 = z * jnp.polyval(GN4, z) / jnp.polyval(GD4, z)

  f8 = jnp.polyval(FN8, z) / (x * jnp.polyval(FD8, z))
  g8 = z * jnp.polyval(GN8, z) / jnp.polyval(GD8, z)

  mask = x < 8.0
  f = jnp.where(mask, f4, f8)
  g = jnp.where(mask, g4, g8)

  si = (np.pi / 2) - f * c - g * s
  ci = f * s - g * c

  return si, ci