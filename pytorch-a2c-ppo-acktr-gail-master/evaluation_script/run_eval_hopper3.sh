for seed in 3
do
python main.py --env-name "Hopper-v2" --algo ppo --use-gae --num-steps 2048 --num-processes 1 --lr 3e-4 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1024000 --use-linear-lr-decay --use-proper-time-limits --expert-algo ppo --gail-batch-size 32 --plt-entropy-coef 0.0 --plr-entropy-coef 0.0 --plt-value-loss-coef 0.5 --plr-value-loss-coef 0.5 --favor-zero-expert-reward --gail --icm --extr-reward-weight 1 --expert-reward-weight 0 --controller-coef 0 --intr-coef 1 --save-date 200227 --log-interval 1 --eval-interval 1 --seed $seed
python main.py --env-name "Hopper-v2" --algo ppo --use-gae --num-steps 2048 --num-processes 1 --lr 3e-4 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1024000 --use-linear-lr-decay --use-proper-time-limits --expert-algo ppo --gail-batch-size 32 --plt-entropy-coef 0.0 --plr-entropy-coef 0.0 --plt-value-loss-coef 0.5 --plr-value-loss-coef 0.5 --favor-zero-expert-reward --gail --icm --extr-reward-weight 1 --expert-reward-weight 0 --controller-coef 2.0 --intr-coef 1 --save-date 200227 --log-interval 1 --eval-interval 1 --seed $seed
python main.py --env-name "Hopper-v2" --algo ppo --use-gae --num-steps 2048 --num-processes 1 --lr 3e-4 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1024000 --use-linear-lr-decay --use-proper-time-limits --expert-algo ppo --gail-batch-size 32 --plt-entropy-coef 0.0 --plr-entropy-coef 0.0 --plt-value-loss-coef 0.5 --plr-value-loss-coef 0.5 --favor-zero-expert-reward --gail --icm --extr-reward-weight 1 --expert-reward-weight 1 --controller-coef 2.0 --intr-coef 1 --save-date 200227 --log-interval 1 --eval-interval 1 --seed $seed
done

