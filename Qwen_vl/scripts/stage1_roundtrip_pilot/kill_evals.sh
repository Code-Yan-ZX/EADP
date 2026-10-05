#!/usr/bin/env bash
# one-shot cleanup + mmvet launch (avoids pkill self-match)
for pid in $(pgrep -f llava_eval_arm_loader); do
  kill -9 $pid 2>/dev/null
done
for pid in $(pgrep -f llava_full_driver); do
  kill -9 $pid 2>/dev/null
done
sleep 2
echo "left: $(pgrep -f 'llava_eval_arm|llava_full_driver' | wc -l)"
exit 0
