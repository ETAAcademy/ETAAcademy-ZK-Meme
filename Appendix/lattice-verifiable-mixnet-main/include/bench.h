#include <string.h>

#ifndef BENCH_H
#define BENCH_H

/*============================================================================*/
/* Macro definitions                                                          */
/*============================================================================*/

/**
 * Number of times each benchmark is ran.
 */
#define BENCH 	100

/**
 * Runs a new benchmark once.
 *
 * @param[in] LABEL			- the label for this benchmark.
 * @param[in] FUNCTION		- the function to benchmark.
 */
#define BENCH_ONCE(LABEL, FUNCTION)											\
	bench_reset();															\
	printf("BENCH: " LABEL "%*c = ", (int)(32 - strlen(LABEL)), ' ');		\
	bench_before();															\
	FUNCTION;																\
	bench_after();															\
	bench_compute(1);														\
	bench_print();															\

/**
 * Runs a new benchmark a small number of times.
 *
 * @param[in] LABEL			- the label for this benchmark.
 * @param[in] FUNCTION		- the function to benchmark.
 */
#define BENCH_SMALL(LABEL, FUNCTION)										\
	bench_reset();															\
	printf("BENCH: " LABEL "%*c = ", (int)(32 - strlen(LABEL)), ' ');		\
	bench_before();															\
	for (int i = 0; i < BENCH; i++)	{										\
		FUNCTION;															\
	}																		\
	bench_after();															\
	bench_compute(BENCH);													\
	bench_print();															\

/**
 * Runs a new benchmark.
 *
 * @param[in] LABEL			- the label for this benchmark.
 */
#define BENCH_BEGIN(LABEL)													\
	bench_reset();															\
	printf("BENCH: " LABEL "%*c = ", (int)(32 - strlen(LABEL)), ' ');		\
	for (int i = 0; i < BENCH; i++)	{										\

/**
 * Prints the mean timing of each execution in nanoseconds.
 */
#define BENCH_END															\
	}																		\
	bench_compute(BENCH * BENCH);											\
	bench_print()															\

/**
 * Measures the time of one execution and adds it to the benchmark total.
 *
 * @param[in] FUNCTION		- the function executed.
 */
#define BENCH_ADD(FUNCTION)													\
	FUNCTION;																\
	bench_before();															\
	for (int j = 0; j < BENCH; j++) {										\
		FUNCTION;															\
	}																		\
	bench_after();															\

/*============================================================================*/
/* Function prototypes                                                        */
/*============================================================================*/

/**
 * Resets the benchmark data.
 *
 * @param[in] label			- the benchmark label.
 */
void bench_reset(void);

/**
 * Measures the time before a benchmark is executed.
 */
void bench_before(void);

/**
 * Measures the time after a benchmark was started and adds it to the total.
 */
void bench_after(void);

/**
 * Computes the mean elapsed time between the start and the end of a benchmark.
 *
 * @param benches			- the number of executed benchmarks.
 */
void bench_compute(int benches);

/**
 * Prints the last benchmark.
 */
void bench_print(void);

/**
 * Returns the result of the last benchmark.
 *
 * @return the last benchmark.
 */
unsigned long long bench_total(void);

#endif /* !BENCH_H */
