"""CPU exact-rational toy certificate; does not inspect or certify GPU kernels."""
from fractions import Fraction

U=Fraction(1,2**24)
MIN_NORMAL=Fraction(1,2**126)
MAX_FINITE=Fraction(2**128-2**104)

def gamma(m):
 if type(m) is not int or m<1 or m*U>=1:raise ValueError('invalid gamma length')
 return m*U/(1-m*U)

def normal_or_zero(x):return x==0 or MIN_NORMAL<=abs(x)<=MAX_FINITE

def round_f32(x):
 """Exact round-to-nearest-even for nonnegative normal/zero toy operands."""
 x=Fraction(x)
 if x<0 or not normal_or_zero(x):raise ValueError('outside toy domain')
 if not x:return x
 e=x.numerator.bit_length()-x.denominator.bit_length()
 step=Fraction(2)**(e-23)
 if x<Fraction(2)**e:e-=1;step/=2
 y=x/step;q,r=divmod(y.numerator,y.denominator)
 if r*2>y.denominator or (r*2==y.denominator and q%2):q+=1
 result=q*step
 if not normal_or_zero(result):raise ValueError('rounded underflow/overflow')
 return result

def toy_gains(similarity,coverage,weights,order):
 """Explicit two sum trees, each rounding every +, product and difference."""
 if not (len(similarity)==len(coverage)==len(weights)==3) or order not in ('author','reassociated','fma'):
  raise ValueError('toy requires exactly three finite rational terms')
 if any(not isinstance(x,Fraction) or x<0 or x>1 or not normal_or_zero(x) for xs in (similarity,coverage,weights) for x in xs):raise ValueError('outside certified input domain')
 d=[round_f32(max(s,c)-c) for s,c in zip(similarity,coverage)]
 if order=='fma':
  x=round_f32(d[1]*weights[1]);x=round_f32(d[0]*weights[0]+x);return round_f32(x+round_f32(d[2]*weights[2]))
 x=[round_f32(a*b) for a,b in zip(d,weights)]
 if order=='author':return round_f32(round_f32(x[0]+x[1])+x[2])
 return round_f32(x[0]+round_f32(x[1]+x[2]))

def certified_gap(best,second,weight_sum,n,*,domain_proven=False):
 """Strict >4γW is conditional on audited normal/zero arithmetic model.

 Arguments must be exact Fractions, i.e. conversion/certificate rounding is
 outside this prototype. Never claim that a boolean proves GPU implementation.
 """
 if not domain_proven:return False
 if not all(isinstance(x,Fraction) for x in (best,second,weight_sum)):raise ValueError('exact rationals required')
 if weight_sum<0 or not 0<=second<=best or weight_sum>n:raise ValueError('invalid conditional model')
 return best-second>4*gamma(n+2)*weight_sum
