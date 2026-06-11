# DAXplain-8B — Evaluation Results

Base **Llama 3.1 8B Instruct** vs **finetuned DAXplain-8B** on 100 held-out DAX measures, scored by Claude Sonnet 4.6 (blind to source) on a 1–5 rubric.

| Metric | Base | Finetuned | Δ |
|---|---|---|---|
| accuracy | 3.11 | 2.91 | -0.20 |
| conciseness | 2.51 | 2.17 | -0.34 |
| bi_fluency | 2.29 | 3.21 | +0.92 |
| overall | 2.64 | 2.76 | +0.13 |

## Worked examples

### `Min Order Value = MINX ( VALUES ( Sales[OrderNumber] ), CALCULATE ( SUMX ( Sales, Sales[Quantity] * Sales[UnitPrice] ) ) )`
*Base scores* {'accuracy': 3, 'conciseness': 2, 'bi_fluency': 2} — *Finetuned scores* {'accuracy': 4, 'conciseness': 3, 'bi_fluency': 4}

**Base:** Let's break down the given DAX measure step by step.

**Measure Name:** Min Order Value

**Purpose:** This measure calculates the minimum order value across all orders. An order value is calculated as the product of quantity and unit price per item sold.

Here's how it works:

1. **MINX**: The `MINX` function returns the smallest value from an expression evaluated over each row that contributes to the aggregation.
2. **VALUES (Sales[OrderNumber])**: This part generates a list of unique order numbers found in the 'Sales' table. Think of it like creating a set of distinct order IDs.
3. **CALCULATE**: This function allows you to evaluate an expression within a specific context or scope. In this case, we're calculating the sum of order values for each order number.
4. **SUMX (Sales, Sales[Quantity] * Sales[UnitPrice])**: Here, we're using the `SUMX` function to calculate the total sales amount for each order. It does this by multiplying the quantity (`Sales[Quantity]`) with the unit price (`Sales[UnitPrice]`) for each item sold in the order.

Now, let's put these pieces together:

- For each unique order number, we calculate its corresponding

**Finetuned:** This measure finds the single lowest total order value across all orders visible in the current filter context - useful for spotting your worst-performing sales transactions. It works by first using VALUES to get each distinct OrderNumber within whatever slicers or filters are active, then iterating over that list with MINX; inside each iteration, CALCULATE shifts the row context into an equivalent filter context so SUMX can correctly sum Quantity × UnitPrice for just that one order before comparing it against other totals. The key watch-out is that if multiple rows share the same OrderNumber but have different line-item quantities and prices, those rows will be summed together under that OrderNumber - meaning you're looking at aggregated order values, not individual line items, which may surprise users who expect a true "minimum line item" figure. Also note that because CALCULATE implicitly adds a filter on OrderNumber during iteration, any existing filter on OrderNumber from outside sources (like a report-level slicer) will stack rather than override the inner VALUES, potentially inflating the number of iterations more than expected. Always validate how OrderNumber is stored and filtered upstream before trusting this as a true minimum revenue metric. 

One common pattern where this measure breaks is when OrderNumber contains duplicates due to data quality issues - say two identical Order

### `Top 5 Sales Reps Profit = CALCULATE ( [Total Profit], TOPN ( 5, ALL ( Sales[SalesRepName] ), [Total Profit], DESC ) )`
*Base scores* {'accuracy': 3, 'conciseness': 2, 'bi_fluency': 2} — *Finetuned scores* {'accuracy': 4, 'conciseness': 3, 'bi_fluency': 4}

**Base:** Let's break down the given DAX measure step by step.

**What does it do?**

This measure calculates the total profit for the top 5 sales representatives with the highest profits. It essentially ranks the sales reps based on their individual profits and returns the sum of profits for the top performers.

**Here's how it works:**

1. **`CALCULATE ([Total Profit])`:**
   - This function is used to calculate an expression (`[Total Profit]`) within a specific context.
   - In this case, we're calculating the `[Total Profit]` measure.

2. **`TOPN (5,...)`:** 
   - `TOPN` is a function that returns the top N values from a table or column.
   - Here, we want the top 5 values, so we pass `5` as the first argument.
   - The second argument is the table or column containing the data we want to rank. We use `ALL(Sales[SalesRepName])`, which means we consider all unique sales rep names without any filters applied.

3. **`..., [Total Profit], DESC)`:** 
   - The third argument specifies the value we want to rank by. In this case, it's the `[

**Finetuned:** This measure returns the total profit contributed by only the top five sales reps - regardless of how many reps are actually present in any given filter context. It works by using TOPN to scan all sales reps visible after stripping away any existing filter on SalesRepName via ALL, then ranking them by their Total Profit from highest to lowest and selecting just those five; CALCULATE finally applies that filtered set back over the same Total Profit measure to produce the result. The key watch-out is that if your data has gaps or nulls in SalesRepName where no transactions exist, those reps will be excluded entirely rather than being ranked at zero - so you may see fewer than five rows returned depending on your data quality. Also note that because TOPN operates within whatever columns are passed as arguments, it's not strictly a "top 5 globally" figure but rather "top 5 among whoever remains after ALL(SalesRepName) is applied," which can behave differently when used inside other measures versus directly in a visual. One common mistake is assuming this always surfaces exactly five reps - if there are ties at the fifth position, TOPN will return more than five rows, potentially inflating the total beyond what users expect. To enforce an exact global top-five list, consider adding a tie-break

### `Max Sale Per Customer = MAXX ( VALUES ( Customer[CustomerKey] ), CALCULATE ( SUMX ( Sales, Sales[Quantity] * Sales[UnitPrice] ) ) )`
*Base scores* {'accuracy': 3, 'conciseness': 2, 'bi_fluency': 2} — *Finetuned scores* {'accuracy': 4, 'conciseness': 3, 'bi_fluency': 4}

**Base:** Let's break down the given DAX measure step by step.

**Measure Name:** Max Sale Per Customer

**Purpose:** This measure calculates the maximum total sales amount per customer across all orders.

Here's how it works:

1. **MAXX function**: The `MAXX` function is used to find the maximum value of an expression within a set of values.
2. **VALUES function**: Inside the `MAXX` function, we have `VALUES(Customer[CustomerKey])`. This returns a table with unique values from the `CustomerKey` column in the `Customers` table.
3. **CALCULATE function**: We use `CALCULATE` to create a new filter context that will be applied to the rest of the calculation.
4. **SUMX function**: Within the `CALCULATE` function, we have `SUMX(Sales,...)`, which means we're summing up the results of another expression over each row in the `Sales` table.
5. **Expression inside SUMX**: The expression `Sales[Quantity] * Sales[UnitPrice]` multiplies the quantity sold (`Quantity`) by the unit price (`UnitPrice`) for each sale.

So, when you put everything together:

- For each unique customer key, calculate

**Finetuned:** This measure finds the single highest revenue transaction amount across all customers visible in the current filter context - useful for spotting your top-spending customer or understanding how much of your total sales is concentrated at the high end. It works by first using VALUES to get each distinct CustomerKey within whatever slicers or filters are active, then iterating over that list with MAXX; inside each iteration, CALCULATE shifts the row context into a filter context so SUMX can correctly sum quantity × unit price for just that one customer's transactions before returning the maximum result. The key watch-out is that if multiple rows in the Sales table belong to the same order but different line items, those rows will be summed together under each customer, meaning you're looking at gross-order totals rather than individual-line values - which may or may not match business expectations depending on how orders and lines are modelled. Also note that because VALUES respects only the Customer dimension column, any other active filters (date range, product category) still apply during the inner CALCULATE, so the "max per customer" figure reflects what those filtered results look like through each customer lens. If you need true max-per-customer regardless of external filters, you'd need to wrap everything in ALLSELECTED(Customer[CustomerKey]) or similar.
