﻿# MC Sim
This Visual Studio project is my coursework for ```CMP 202``` at Abertay University, Scotland.

## underlying concept
```
dS = μ S dt + σ S ds

S = stock price
μ = exp. return
σ = volatility
```
## approach
```
repeat n times
  repeat t times
    generate normal distributed number
    update end price
  save end price to path array  
sort path array in descending order
extract the nth quantile
scale VAR to holding period using square root of time
print results
```
## boundary
- Assuming a stock without dividend payments
- single stock portfolio
- using C++ AMP
- optimized for a NVIDIA GeForce 940MX

## Todos
- [x] Prepare proposal
- [ ] Implement MC kernel
- [ ] Implement path array to value at risk functionality
- [ ] Optimize memory access, loop unrolling etc.
- [ ] measure performance
- [ ] Prepare presentation
- [ ] Enhance to a multi stock portfolio (optional)
- [ ] Download historical data from quandl (optional)
- [ ] Calculate volatility using EWMA method (optional)
## Contact
Feel free to send me a  [mail](mailto:1705042@abertay.ac.uk)