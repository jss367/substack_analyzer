**Notes**

- Daily is preferred. If only monthly data are available, run a monthly mode with equivalent likelihoods (e.g., NegBin on monthly adds, month dummies for seasonality). Clearly tag imputed/resampled periods (e.g., `is_imputed`) and propagate this into larger observation noise during fitting to reflect reduced information.

Dropping Seasonality

If you don’t think there’s a strong day-of-week or monthly cycle in Substack growth, you can drop all the Fourier terms and DOW dummies. The trade-off: you’ll lose the ability to explain small wiggles like “people unsubscribe more on Mondays” or “signups spike on weekends.” But for most Substacks, the dominant signal is events + ads, not subtle weekly cycles. So removing seasonality is a safe simplification.

should events have end dates?
