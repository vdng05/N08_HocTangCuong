"""Train and evaluate on-policy first-visit Monte Carlo control."""

import argparse
import sys

import config
from algorithms.monte_carlo import OnPolicyFirstVisitMC
from environment.delivery_env import DeliveryGridEnv


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Huấn luyện On-policy First-visit Monte Carlo cho môi trường giao hàng."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10_000,
        help="Số episode huấn luyện (mặc định: 10000).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Xác suất khám phá epsilon (mặc định: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed để tái lập kết quả (mặc định: 42).",
    )
    args = parser.parse_args()

    env = DeliveryGridEnv(
        random_customers=True,
        min_customers=3,
        max_customers=5,
    )
    solver = OnPolicyFirstVisitMC(
        env=env,
        gamma=config.GAMMA,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    history = solver.train(
        num_episodes=args.episodes,
        verbose=True,
        log_interval=max(1, args.episodes // 10),
    )
    route = solver.extract_greedy_route()

    recent_rewards = history["episode_rewards"][-min(1_000, args.episodes) :]
    print("\nKẾT QUẢ ON-POLICY FIRST-VISIT MONTE CARLO")
    print(f"  Số episode:                  {args.episodes}")
    print(f"  Gamma:                       {config.GAMMA}")
    print(f"  Epsilon:                     {args.epsilon}")
    print("  Số điểm giao ngẫu nhiên:     3–5")
    print(f"  Tỷ lệ hoàn thành episode:    {history['completion_rate']:.2%}")
    print(f"  Thưởng trung bình gần nhất:  {sum(recent_rewards) / len(recent_rewards):.2f}")
    print(f"  Thời gian huấn luyện:        {history['training_time_seconds']:.2f} giây")
    print(
        "  Greedy route hoàn thành:    "
        f"{'Có' if route['is_completed'] else 'Chưa'}"
    )
    print(f"  Số bước greedy route:        {route['total_steps']}")
    print(f"  Reward greedy route:         {route['total_reward']:.2f}")
    print(f"  Tọa độ khách hàng lần cuối:  {route['customer_positions']}")
    print(
        "  Thứ tự giao hàng:           "
        + (
            " -> ".join(
                item["customer_id"] for item in route["visited_customers_order"]
            )
            or "Chưa giao được khách nào"
        )
    )


if __name__ == "__main__":
    main()
