#!/usr/bin/env python3
"""
Batch process images using FADE algorithm with parallel workers.
Processes images in batches to handle millions of images efficiently.
"""

import sys
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import time
import tempfile
import sys
from PIL import Image

# make sure the repository's `src` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import login as netlogin
from src import fetch_image as netfetch


class FADEProcessor:
    def __init__(self, fade_dir, output_dir='data', batch_size=10000, num_workers=None):
        """
        Initialize FADE processor.
        
        Args:
            fade_dir: Directory containing FADE MATLAB files
            output_dir: Directory to save parquet results
            batch_size: Number of images to process before saving to parquet
            num_workers: Number of parallel workers (default: CPU count)
        """
        self.fade_dir = Path(fade_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.batch_size = batch_size
        self.num_workers = num_workers or cpu_count()
        self.octave_script = self.fade_dir / 'process_single_image.m'
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify octave script exists
        if not self.octave_script.exists():
            raise FileNotFoundError(f"Octave script not found: {self.octave_script}")
        
        print(f"FADE directory: {self.fade_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Using {self.num_workers} worker processes")
    
    @staticmethod
    def process_single_image(args):
        """
        Static method to process a single image (for multiprocessing).
        
        Args:
            args: Tuple of (image_path, fade_dir, octave_script)
            
        Returns:
            dict: Result with image info and score
        """
        image_path, fade_dir, octave_script = args
        
        try:
            # Run octave script with image path
            result = subprocess.run(
                ['octave-cli', str(octave_script), str(image_path)],
                cwd=str(fade_dir),
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout per image
            )
            
            if result.returncode == 0:
                # Parse density score from output
                try:
                    score = float(result.stdout.strip())
                    return {
                        'image_id': str(Path(image_path).name),
                        'image_path': str(image_path),
                        'score': score,
                        'success': True
                    }
                except ValueError:
                    return {
                        'image_path': str(image_path), 
                        'success': False, 
                        'error': f'invalid_output: stdout="{result.stdout[:100]}" stderr="{result.stderr[:100]}"'
                    }
            else:
                return {
                    'image_path': str(image_path), 
                    'success': False, 
                    'error': f'octave_error (code {result.returncode}): stdout="{result.stdout[:100]}" stderr="{result.stderr[:200]}"'
                }
                
        except subprocess.TimeoutExpired:
            return {'image_path': str(image_path), 'success': False, 'error': 'timeout'}
        except Exception as e:
            return {'image_path': str(image_path), 'success': False, 'error': str(e)}
    
    def process_batch_parallel(self, image_paths, batch_num):
        """
        Process a batch of images in parallel and save to parquet.
        
        Args:
            image_paths: List of image paths to process
            batch_num: Batch number for output filename
        """
        start_time = time.time()
        
        # Prepare arguments for parallel processing (convert to absolute paths)
        args_list = [(str(Path(img).resolve()), self.fade_dir, self.octave_script) for img in image_paths]
        
        # Process images in parallel
        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(self.process_single_image, args_list),
                total=len(image_paths),
                desc=f"Batch {batch_num}"
            ))
        
        # Filter successful results
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        # Save successful results to parquet
        if successful_results:
            df = pd.DataFrame([{
                'image_id': r['image_id'],
                'image_path': r['image_path'],
                'score': r['score']
            } for r in successful_results])
            
            output_file = self.output_dir / f'fade_results_batch_{batch_num:05d}.parquet'
            df.to_parquet(output_file, index=False, engine='pyarrow')
            
            elapsed = time.time() - start_time
            images_per_sec = len(successful_results) / elapsed if elapsed > 0 else 0
            
            print(f"✓ Batch {batch_num}: {len(successful_results)}/{len(image_paths)} successful "
                  f"({images_per_sec:.1f} img/s)")
            print(f"  Saved to: {output_file}")
            
            # Save failed results log if any
            if failed_results:
                failed_file = self.output_dir / f'failed_batch_{batch_num:05d}.txt'
                with open(failed_file, 'w') as f:
                    for r in failed_results:
                        f.write(f"{r['image_path']}: {r.get('error', 'unknown')}\n")
                print(f"  ⚠ {len(failed_results)} failed (logged to {failed_file})")
            
            return len(successful_results)
        else:
            print(f"✗ Batch {batch_num}: No successful results!")
            if failed_results:
                print(f"  All {len(failed_results)} images failed. Check errors:")
                for r in failed_results[:3]:  # Show first 3 errors
                    print(f"    {r.get('error', 'unknown')}")
        return 0
    
    def process_images_by_id_range(self, start_id, end_id, sid=None, matlab_bin='matlab', gpu_index=None, matlab_timeout=600):
        """
        Fetch images by numeric IDs from network storage using `src.login` and `src.fetch_image`,
        batch them to local temp files, and process each batch via the MATLAB batch processor.

        Args:
            start_id: starting numeric ID (inclusive)
            end_id: ending numeric ID (inclusive)
            sid: optional session id; if None the function will call `src.login.login()`
            matlab_bin: path to matlab binary
            gpu_index: GPU index to pass to MATLAB (None -> no explicit selection)
            matlab_timeout: timeout per MATLAB batch in seconds
        """
        if sid is None:
            sid = netlogin.login()
            if not sid:
                print("Unable to obtain session id; aborting")
                return

        total = int(end_id) - int(start_id) + 1
        print(f"Processing IDs {start_id}..{end_id} ({total} images) in batches of {self.batch_size}")

        batch_num = 0
        overall_start = time.time()
        total_processed = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            current_paths = []

            for i in range(int(start_id), int(end_id) + 1):
                try:
                    img = netfetch.process_image(i, sid)
                except Exception as e:
                    print(f"Fetch error for id {i}: {e}")
                    img = None

                if img is None:
                    # record failed fetch
                    failed_file = self.output_dir / f'failed_fetch_{i:07d}.txt'
                    with open(failed_file, 'w') as f:
                        f.write('fetch_failed\n')
                    continue

                local_path = tmpdir / f'sa_{i}.jpg'
                try:
                    if isinstance(img, Image.Image):
                        img.save(local_path)
                    else:
                        # if fetch_image returns bytes
                        with open(local_path, 'wb') as wf:
                            wf.write(img)
                except Exception as e:
                    print(f"Failed to save image {i}: {e}")
                    continue

                current_paths.append(str(local_path))

                if len(current_paths) >= self.batch_size:
                    batch_num += 1
                    processed = self.process_batch_matlab(current_paths, batch_num, matlab_bin=matlab_bin, gpu_index=gpu_index, timeout=matlab_timeout)
                    total_processed += processed

                    # cleanup
                    for p in current_paths:
                        try:
                            Path(p).unlink()
                        except Exception:
                            pass
                    current_paths = []

            if current_paths:
                batch_num += 1
                processed = self.process_batch_matlab(current_paths, batch_num, matlab_bin=matlab_bin, gpu_index=gpu_index, timeout=matlab_timeout)
                total_processed += processed
                for p in current_paths:
                    try:
                        Path(p).unlink()
                    except Exception:
                        pass

        elapsed = time.time() - overall_start
        avg_speed = total_processed / elapsed if elapsed > 0 else 0
        print(f"\n{'='*60}")
        print(f"✓ Processing complete!")
        print(f"Total processed: {total_processed}/{total}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Average speed: {avg_speed:.1f} images/second")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}")

    def process_batch_matlab(self, image_paths, batch_num, matlab_bin='matlab', gpu_index=None, timeout=600):
        """
        Process a batch of images by calling a MATLAB wrapper that processes many images in one MATLAB invocation.

        This reduces per-image process overhead and allows MATLAB to select and use a GPU if the
        MATLAB wrapper calls `gpuDevice(gpu_index)`.
        """
        start_time = time.time()

        # Create temp file with image paths
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tf:
            list_path = Path(tf.name)
            for p in image_paths:
                tf.write(str(Path(p).resolve()) + '\n')

        # Output parquet that MATLAB will write
        out_file = self.output_dir / f'fade_results_batch_{batch_num:05d}.parquet'

        # Prepare MATLAB-safe quoted strings (single quotes doubled)
        def mquote(s):
            return "'" + str(s).replace("'", "''") + "'"

        gpu_arg = '[]' if gpu_index is None else str(int(gpu_index))

        # Use -batch if available (MATLAB R2019a+). This will run the function and exit.
        matlab_cmd = [str(matlab_bin), '-batch', f"process_batch_from_file({mquote(str(list_path))}, {mquote(str(out_file))}, {gpu_arg})"]

        try:
            proc = subprocess.run(
                matlab_cmd,
                cwd=str(self.fade_dir),
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if proc.returncode != 0:
                print(f"MATLAB batch failed (code {proc.returncode}): {proc.stderr[:200]}")
                return 0

            # Read parquet results written by MATLAB
            results = []
            if out_file.exists():
                try:
                    df = pd.read_parquet(out_file)
                    for _, row in df.iterrows():
                        try:
                            score = float(row.get('score', float('nan')))
                            img_path = row.get('image_path')
                            results.append({
                                'image_id': Path(img_path).name,
                                'image_path': img_path,
                                'score': score,
                                'success': not (score!=score)
                            })
                        except Exception:
                            results.append({'image_path': row.get('image_path', None), 'success': False, 'error': 'parse_error'})
                except Exception as e:
                    print(f"Failed to read parquet output {out_file}: {e}")
                    return 0
            else:
                print(f"Expected MATLAB output {out_file} not found")
                return 0

            # Convert to parquet and cleanup CSV
            successful_results = [r for r in results if r.get('success', False)]
            failed_results = [r for r in results if not r.get('success', False)]

            if successful_results:
                df = pd.DataFrame([{
                    'image_id': r['image_id'],
                    'image_path': r['image_path'],
                    'score': r['score']
                } for r in successful_results])

                output_file = self.output_dir / f'fade_results_batch_{batch_num:05d}.parquet'
                df.to_parquet(output_file, index=False, engine='pyarrow')

                elapsed = time.time() - start_time
                images_per_sec = len(successful_results) / elapsed if elapsed > 0 else 0

                print(f"✓ Batch {batch_num}: {len(successful_results)}/{len(image_paths)} successful "
                      f"({images_per_sec:.1f} img/s)")
                print(f"  Saved to: {output_file}")

                if failed_results:
                    failed_file = self.output_dir / f'failed_batch_{batch_num:05d}.txt'
                    with open(failed_file, 'w') as f:
                        for r in failed_results:
                            f.write(f"{r.get('image_path')}: {r.get('error', 'unknown')}\n")
                    print(f"  ⚠ {len(failed_results)} failed (logged to {failed_file})")

                # remove parquet
                try:
                    out_file.unlink()
                except Exception:
                    pass

                # remove temp list
                try:
                    list_path.unlink()
                except Exception:
                    pass

                return len(successful_results)
            else:
                print(f"✗ Batch {batch_num}: No successful results!")
                for r in failed_results[:3]:
                    print(f"    {r.get('error', 'unknown')}")
        except subprocess.TimeoutExpired:
            print(f"MATLAB batch timed out for batch {batch_num}")
        finally:
            # best-effort cleanup
            try:
                if list_path.exists():
                    list_path.unlink()
            except Exception:
                pass

        return 0
    
    def process_images_from_list(self, image_list_file):
        """
        Process images from a text file containing image paths (one per line).
        
        Args:
            image_list_file: Path to text file with image paths
        """
        # Read image paths
        with open(image_list_file, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        total_images = len(image_paths)
        print(f"Found {total_images} images to process")
        print(f"Processing in batches of {self.batch_size} with {self.num_workers} workers\n")
        
        # Process in batches
        batch_num = 0
        total_processed = 0
        overall_start = time.time()
        
        for i in range(0, total_images, self.batch_size):
            batch = image_paths[i:i + self.batch_size]
            batch_num += 1
            processed = self.process_batch_parallel(batch, batch_num)
            total_processed += processed
        
        elapsed = time.time() - overall_start
        avg_speed = total_processed / elapsed if elapsed > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"✓ Processing complete!")
        print(f"Total processed: {total_processed}/{total_images}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Average speed: {avg_speed:.1f} images/second")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}")
    
    def process_images_from_directory(self, image_dir, extensions=None):
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory containing images
            extensions: List of image extensions to process (default: common image formats)
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        image_dir = Path(image_dir)
        
        # Find all images
        print("Scanning for images...")
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.rglob(f'*{ext}'))
        
        total_images = len(image_paths)
        print(f"Found {total_images} images in {image_dir}")
        print(f"Processing in batches of {self.batch_size} with {self.num_workers} workers\n")
        
        # Process in batches
        batch_num = 0
        total_processed = 0
        overall_start = time.time()
        
        # image_paths = image_paths[:1000]  # --- IGNORE ---

        for i in range(0, total_images, self.batch_size):
            batch = image_paths[i:i + self.batch_size]
            batch_num += 1
            processed = self.process_batch_parallel(batch, batch_num)
            total_processed += processed
        
        elapsed = time.time() - overall_start
        avg_speed = total_processed / elapsed if elapsed > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"✓ Processing complete!")
        print(f"Total processed: {total_processed}/{total_images}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Average speed: {avg_speed:.1f} images/second")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}")


def merge_parquet_files(data_dir, output_file='fade_results_complete.parquet'):
    """
    Merge all batch parquet files into a single file.
    
    Args:
        data_dir: Directory containing batch parquet files
        output_file: Name of merged output file
    """
    data_dir = Path(data_dir)
    batch_files = sorted(data_dir.glob('fade_results_batch_*.parquet'))
    
    if not batch_files:
        print("No batch files found to merge")
        return
    
    print(f"Merging {len(batch_files)} batch files...")
    
    dfs = []
    for batch_file in tqdm(batch_files, desc="Reading batches"):
        dfs.append(pd.read_parquet(batch_file))
    
    # Concatenate all dataframes
    df_complete = pd.concat(dfs, ignore_index=True)
    
    # Save merged file
    output_path = data_dir / output_file
    df_complete.to_parquet(output_path, index=False, engine='pyarrow')
    
    print(f"\n✓ Merged {len(df_complete)} records to {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    
    # Show statistics
    print(f"\nScore statistics:")
    print(df_complete['score'].describe())
    
    # Optionally remove batch files
    response = input("\nDelete batch files? (y/n): ")
    if response.lower() == 'y':
        for batch_file in batch_files:
            batch_file.unlink()
        print("✓ Batch files deleted")


def main():
    parser = argparse.ArgumentParser(description='Batch process images using FADE algorithm with parallel workers')
    parser.add_argument('--fade-dir', default='FADE', help='Directory containing FADE MATLAB files')
    parser.add_argument('--output-dir', default='data', help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--workers', type=int, help=f'Number of parallel workers (default: {cpu_count()} CPUs)')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image-list', help='Text file with image paths (one per line)')
    group.add_argument('--image-dir', help='Directory containing images to process')
    
    parser.add_argument('--merge', action='store_true', help='Merge batch parquet files after processing')
    parser.add_argument('--merge-only', action='store_true', help='Only merge existing batch files')
    
    args = parser.parse_args()
    
    if args.merge_only:
        merge_parquet_files(args.output_dir)
        return
    
    # Initialize processor
    processor = FADEProcessor(
        fade_dir=args.fade_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    
    # Process images
    if args.image_list:
        processor.process_images_from_list(args.image_list)
    elif args.image_dir:
        processor.process_images_from_directory(args.image_dir)
    
    # Merge if requested
    if args.merge:
        merge_parquet_files(args.output_dir)


if __name__ == '__main__':
    main()