# GA Startup Time Investigation

Note: this note answers the question, "What is the logic flow during initial setup, before generation timing begins, and why is that startup taking so long?" It is about startup only, not about the time each generation takes after that.

It is not doing host fitness evaluation, and it is not generating generation 0 on device.

What it actually does is:

- Load the word list on host and upload that small catalog to device constant memory. That part is cheap.
- Compute budgets and sizes, then generate the entire initial population on the CPU. The actual work is: allocate one huge host buffer, `memset` it to zero, then single-threadedly fill every genome with random weights and random output tails.
- Copy that host population into a second huge host structure, the genotype slab. Creating the host slab does another full `memset` of the slab. Then for each individual it allocates a slab slot, zeroes that slot again, and copies the full genome bytes into it.
- Allocate the full device slab runtime. That allocates device buffers for the whole slab and zeroes the entire device slab.
- Upload the full host slab to the GPU.
- Only after all of that does it print the startup banner.

So for the original run, startup is roughly:

- Build `28224` genomes on host.
- Hold about `7.95 GiB` of host population bytes.
- Allocate and zero about `12.0 GiB` of host slab bytes.
- Copy the live genomes from the host population into that host slab.
- Allocate and zero about `12.0 GiB` of device slab bytes.
- Copy about `12.0 GiB` over PCIe to the device.

Why it is so slow:

- The biggest likely offender is CPU random initialization. It is single-threaded, and it uses `std::normal_distribution` for every weight and tail value. With the current layer sizes, that comes out to about `150,648` Gaussian draws per genome, or about `4.25 billion` Gaussian draws total for `28,224` genomes.
- Then it does a lot of redundant memory traffic on host. It zeroes the host population buffer, fills it, zeroes the entire host slab, zeroes each live slab slot again, and copies all live genomes into the slab.
- Then it zeroes the entire device slab and immediately overwrites it with a full-slab host-to-device copy.
- The host `population` is not freed after it is copied into the slab, so during startup it keeps both the `~8 GiB` population and the `~12 GiB` slab resident in host RAM. If the machine is even a little memory-tight, that can turn slow into catastrophic.

One important boundary:

- There is no host fitness work in startup.
- The first fitness evaluation is on device.
- If the long silent period is before the `Running device GA demo ...` line, that is pure startup.
- If the long silent period is before the first `Generation 0:` line, that includes startup plus the first device fitness and evolution pass.

So the short answer is: it is doing the worst of both worlds for startup. It generates generation 0 on host, materializes it again into a second host representation, allocates and clears a full device slab, and uploads the full slab. The most likely dominant cost is the single-threaded CPU random initialization, with full-slab host and device memory traffic as the second big cost.
