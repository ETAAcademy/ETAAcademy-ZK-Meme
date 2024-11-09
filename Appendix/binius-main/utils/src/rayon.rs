// Copyright 2024 Irreducible Inc.

use std::{env, sync::OnceLock};

/// In case when number of threads is set to 1, use rayon thread pool with
/// `use_current_thread` set to true. This is solves two problems:
/// 1. The performance is almost the same as if rayon wasn't used at all
/// 2. Makes profiling and debugging results less noisy
///
/// NOTE: rayon doesn't allow initializing global thread pool several times, so
/// in case when it was initialized before this function returns an error.
/// The typical usage of the function is to place it's call in the beginning of the `main`.
/// The function returns reference to the result because `ThreadPoolBuildError`
/// doesn't implement `Clone`.
pub fn adjust_thread_pool() -> &'static Result<(), rayon::ThreadPoolBuildError> {
	static ONCE_GUARD: OnceLock<Result<(), rayon::ThreadPoolBuildError>> = OnceLock::new();

	ONCE_GUARD.get_or_init(|| {
		// We cannot use `rayon::get_current_threads` because it would force the global thread pool
		// to initialize, so we won't be able to override it.
		match env::var("RAYON_NUM_THREADS") {
			Ok(v) if v == "1" => rayon::ThreadPoolBuilder::new()
				.num_threads(1)
				.use_current_thread()
				.build_global(),
			_ => Ok(()),
		}
	})
}

/// Returns the base-2 logarithm of the number of threads that should be used for the task
pub fn get_log_max_threads() -> usize {
	(2 * rayon::current_num_threads() - 1).ilog2() as _
}
