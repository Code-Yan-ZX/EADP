import itertools, unittest
from decimal import Decimal
from analyze import optimize

class TestOptimum(unittest.TestCase):
    def test_matches_exhaustive_assignments(self):
        scores=[{128:Decimal('0.2'),256:Decimal('0.7'),512:Decimal('1')},
                {128:Decimal('0.6'),256:Decimal('0.5'),512:Decimal('0.9')},
                {128:Decimal('0.4'),256:Decimal('0.5'),512:Decimal('0.7')},
                {128:Decimal('0'),256:Decimal('1'),512:Decimal('0.8')}]
        for capacity in range(4,13):
            value,cost,ks=optimize(scores,capacity)
            possible=[(sum((scores[i][k] for i,k in enumerate(v)),Decimal(0)),sum(k//128 for k in v)) for v in itertools.product((128,256,512),repeat=len(scores)) if sum(k//128 for k in v)<=capacity]
            best=max(v for v,c in possible)
            self.assertEqual(value,best);self.assertEqual(cost,min(c for v,c in possible if v==best))
            self.assertEqual(sum(scores[i][k] for i,k in enumerate(ks)),value)

if __name__=='__main__':unittest.main()
