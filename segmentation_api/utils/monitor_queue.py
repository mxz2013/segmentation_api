import redis
from rq import Queue, Worker
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def monitor_queue():
    redis_conn = redis.Redis(host="localhost", port=6379)
    queue = Queue("ml_inference_queue", connection=redis_conn)

    print("🔄 Monitoring RQ Queue... (Ctrl+C to stop)")
    print("=" * 50)

    try:
        while True:
            # Get queue statistics
            jobs = queue.jobs
            started_jobs = queue.started_job_registry.get_job_ids()
            finished_jobs = queue.finished_job_registry.get_job_ids()
            failed_jobs = queue.failed_job_registry.get_job_ids()
            # Get workers for this queue
            workers = Worker.all(connection=redis_conn)
            queue_workers = [w for w in workers if queue.name in w.queue_names()]

            print(f"\n📊 Queue Status at {time.strftime('%H:%M:%S')}")
            print(f"⏳ Queued Jobs: {len(jobs)}")
            print(f"🔄 Started Jobs: {len(started_jobs)}")
            print(f"✅ Finished Jobs: {len(finished_jobs)}")
            print(f"❌ Failed Jobs: {len(failed_jobs)}")
            print(f"👥 Workers: {len(queue_workers)}")

            if jobs:
                print("\n📋 Queued Job IDs:")
                for i, job in enumerate(jobs[:5]):  # Show first 5
                    print(f"  {i+1}. {job.id}")
                if len(jobs) > 5:
                    print(f"  ... and {len(jobs) - 5} more")

            time.sleep(2)  # Update every 2 seconds

    except KeyboardInterrupt:
        print("\n👋 Stopping monitor...")


if __name__ == "__main__":
    monitor_queue()
