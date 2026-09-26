Simulate the real datapath: INT8×INT8 → wide acc → bias add → requant shift → saturate. Accumulator bits ≠ bias bits. Don't rely on fake-quant where it diverges from HW.
