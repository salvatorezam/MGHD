#!/usr/bin/env python3
"""
Real-time Quantum Error Correction Decoding Service
Persistent fastpath decoder with streaming syndrome processing
"""

import argparse
import time
import signal
import sys
import threading
import queue
import numpy as np
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

class RealtimeDecodeService:
    """Real-time syndrome decoding service using persistent fastpath LUT."""
    
    def __init__(self, capacity=4096, max_queue_size=1000):
        self.capacity = capacity
        self.max_queue_size = max_queue_size
        self.running = False
        self.stats = {
            'total_decoded': 0,
            'total_batches': 0,
            'total_time': 0.0,
            'max_batch_size': 0,
            'min_latency': float('inf'),
            'max_latency': 0.0
        }
        
        # Threading components
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.fastpath_svc = None
        
    def initialize_fastpath(self):
        """Initialize the persistent fastpath decoder."""
        try:
            import fastpath
            print("Loading rotated d=3 LUT...")
            lut16, Hx, Hz, meta = fastpath.load_rotated_d3_lut_npz()
            
            print(f"Initializing persistent LUT (capacity={self.capacity})...")
            self.fastpath_svc = fastpath.PersistentLUT(lut16=lut16, capacity=self.capacity)
            self.fastpath_svc.__enter__()
            
            print("✅ Fastpath decoder online")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize fastpath: {e}")
            return False
    
    def decode_worker(self):
        """Main worker thread for syndrome processing."""
        print("🚀 Decode worker started")
        
        while self.running:
            try:
                # Get syndrome batch from queue (timeout to check running flag)
                try:
                    batch_data = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                syndrome_batch, timestamp, batch_id = batch_data
                
                # Decode the batch
                start_time = time.perf_counter()
                corrections = self.fastpath_svc.decode_batch(syndrome_batch)
                decode_time = time.perf_counter() - start_time
                
                # Update statistics
                batch_size = len(syndrome_batch)
                self.stats['total_decoded'] += batch_size
                self.stats['total_batches'] += 1
                self.stats['total_time'] += decode_time
                self.stats['max_batch_size'] = max(self.stats['max_batch_size'], batch_size)
                
                latency_us = decode_time * 1e6
                self.stats['min_latency'] = min(self.stats['min_latency'], latency_us)
                self.stats['max_latency'] = max(self.stats['max_latency'], latency_us)
                
                # Put result in output queue
                result = {
                    'batch_id': batch_id,
                    'corrections': corrections,
                    'timestamp': timestamp,
                    'decode_time': decode_time,
                    'batch_size': batch_size
                }
                
                try:
                    self.output_queue.put(result, timeout=0.1)
                except queue.Full:
                    print("⚠️  Output queue full, dropping result")
                
                self.input_queue.task_done()
                
            except Exception as e:
                print(f"❌ Decode worker error: {e}")
                if self.running:
                    time.sleep(0.01)  # Brief pause before retrying
    
    def submit_syndrome_batch(self, syndromes, batch_id=None):
        """Submit a batch of syndromes for decoding."""
        if not self.running:
            raise RuntimeError("Service not running")
        
        if batch_id is None:
            batch_id = f"batch_{int(time.time() * 1000000)}"
        
        # Ensure syndromes are uint8
        if syndromes.dtype != np.uint8:
            syndromes = syndromes.astype(np.uint8)
        
        batch_data = (syndromes, time.perf_counter(), batch_id)
        
        try:
            self.input_queue.put(batch_data, timeout=0.1)
            return batch_id
        except queue.Full:
            raise RuntimeError("Input queue full, cannot accept more syndromes")
    
    def get_result(self, timeout=1.0):
        """Get a decode result from the output queue."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def start(self):
        """Start the decoding service."""
        if self.running:
            return
        
        if not self.initialize_fastpath():
            return False
        
        self.running = True
        self.worker_thread = threading.Thread(target=self.decode_worker, daemon=True)
        self.worker_thread.start()
        
        print(f"🟢 Real-time decode service started")
        return True
    
    def stop(self):
        """Stop the decoding service."""
        if not self.running:
            return
        
        print("🔴 Stopping decode service...")
        self.running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        
        if self.fastpath_svc:
            try:
                self.fastpath_svc.__exit__(None, None, None)
            except:
                pass
        
        self.print_stats()
        print("✅ Service stopped")
    
    def print_stats(self):
        """Print service statistics."""
        if self.stats['total_batches'] == 0:
            print("📊 No batches processed")
            return
        
        avg_batch_size = self.stats['total_decoded'] / self.stats['total_batches']
        avg_latency_us = (self.stats['total_time'] / self.stats['total_batches']) * 1e6
        throughput = self.stats['total_decoded'] / self.stats['total_time'] if self.stats['total_time'] > 0 else 0
        
        print("📊 Service Statistics:")
        print(f"   Total decoded: {self.stats['total_decoded']} syndromes")
        print(f"   Total batches: {self.stats['total_batches']}")
        print(f"   Avg batch size: {avg_batch_size:.1f}")
        print(f"   Max batch size: {self.stats['max_batch_size']}")
        print(f"   Avg latency: {avg_latency_us:.1f}µs")
        print(f"   Min latency: {self.stats['min_latency']:.1f}µs")
        print(f"   Max latency: {self.stats['max_latency']:.1f}µs")
        print(f"   Throughput: {throughput:.0f} syndromes/sec")

def demo_mode(service, duration=10):
    """Run a demonstration of the real-time service."""
    print(f"\n🎬 Running demo for {duration} seconds...")
    
    np.random.seed(42)  # Reproducible demo
    demo_start = time.time()
    batch_id_counter = 0
    submitted_batches = {}
    
    while time.time() - demo_start < duration:
        # Generate random syndrome batch
        batch_size = np.random.randint(1, 64)
        syndromes = np.random.randint(0, 256, size=(batch_size, 8), dtype=np.uint8)
        
        # Submit for decoding
        try:
            batch_id = f"demo_{batch_id_counter}"
            service.submit_syndrome_batch(syndromes, batch_id)
            submitted_batches[batch_id] = (time.time(), batch_size)
            batch_id_counter += 1
            
        except RuntimeError as e:
            print(f"⚠️  Submission failed: {e}")
        
        # Collect results
        result = service.get_result(timeout=0.01)
        if result:
            batch_id = result['batch_id']
            if batch_id in submitted_batches:
                submit_time, _ = submitted_batches[batch_id]
                end_to_end_time = time.time() - submit_time
                print(f"✅ {batch_id}: {result['batch_size']} syndromes, "
                      f"decode={result['decode_time']*1e6:.1f}µs, "
                      f"e2e={end_to_end_time*1e6:.1f}µs")
                del submitted_batches[batch_id]
        
        # Rate limiting
        time.sleep(0.01)
    
    # Collect remaining results
    print("🔄 Collecting remaining results...")
    timeout_start = time.time()
    while submitted_batches and time.time() - timeout_start < 2.0:
        result = service.get_result(timeout=0.1)
        if result and result['batch_id'] in submitted_batches:
            del submitted_batches[result['batch_id']]
    
    if submitted_batches:
        print(f"⚠️  {len(submitted_batches)} batches not returned")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-time Quantum Error Correction Decoding Service")
    parser.add_argument('--capacity', type=int, default=4096, 
                       help='Fastpath ring buffer capacity')
    parser.add_argument('--queue-size', type=int, default=1000,
                       help='Maximum input/output queue size')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration mode')
    parser.add_argument('--demo-duration', type=int, default=10,
                       help='Demo duration in seconds')
    
    args = parser.parse_args()
    
    # Create service
    service = RealtimeDecodeService(capacity=args.capacity, max_queue_size=args.queue_size)
    
    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Shutdown signal received")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start service
    if not service.start():
        print("❌ Failed to start service")
        sys.exit(1)
    
    try:
        if args.demo:
            demo_mode(service, args.demo_duration)
        else:
            print("\n🎯 Service ready for syndrome processing")
            print("📝 Use submit_syndrome_batch() and get_result() methods")
            print("⏹️  Press Ctrl+C to stop")
            
            # Keep alive until interrupted
            while True:
                time.sleep(1)
                
    finally:
        service.stop()

if __name__ == "__main__":
    main()
