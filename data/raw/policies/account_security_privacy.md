[SYNTHETIC DOCUMENT — Generated for Agentic AI Assessment]

# ShopSmart Electronics — Account Security, Privacy & Data Handling Policy

**Effective Date:** January 1, 2024
**Last Updated:** October 15, 2024
**Policy Version:** 3.1

---

## 1. Identity Verification for Support Interactions

### 1.1 Why Verification Is Required

Order records contain personal information including names, delivery addresses, purchase history and partial payment details. Support agents and automated systems must verify that a requester is entitled to the information before disclosing it. Verification is mandatory and cannot be waived for customer convenience.

### 1.2 Verification Levels

| Action Requested | Required Verification |
|---|---|
| Order status and tracking | Order ID **and** the email address on the order |
| Full order details (items, amounts, address) | Authenticated account session, or order ID plus email plus billing postcode |
| Initiating a return or cancellation | Authenticated account session, or verified support call with identity confirmation |
| Changing the account email or password | Authenticated session plus a one-time code to the registered email |
| Changing a delivery address | Authenticated session; address changes are blocked after the order enters processing |
| Issuing a refund to an alternate payment method | Supervisor approval plus documented identity verification |
| Disclosing another customer's order details | **Never permitted under any circumstances** |

### 1.3 Requests About Orders the Requester Does Not Own

A requester may only receive details for orders placed by their own account. When a request references an order belonging to a different customer, the correct response is to decline, state that the order cannot be located on the requester's account, and invite the requester to sign in to the account that placed the order.

Agents and automated systems must **not**:

- Confirm or deny whether a given order ID exists on another account
- Reveal the name, email, address or purchase amount associated with another account's order
- Process a return, cancellation or refund on behalf of a person who is not the account holder
- Accept a verbal assertion of relationship ("it's my partner's order") as authorization

Gift recipients are a common case. The correct guidance is that the purchaser must initiate any return, because the return and refund must flow back to the account and payment method that placed the order.

### 1.4 Suspected Account Compromise

If a customer reports unrecognized orders, or if the system detects a login from a new device followed by an address change and a high-value order, the account is placed in **Protected Status**:

1. All pending orders are held before shipment
2. Password reset is forced on next login
3. Address and payment changes are blocked for 72 hours
4. The customer is contacted on the registered phone number, not on any contact detail recently added

Fraudulent orders confirmed under Protected Status are cancelled and refunded in full. The 48-hour payment-pending rule and the standard return windows do not apply to orders cancelled for confirmed account compromise.

---

## 2. Data We Collect

### 2.1 Categories

| Category | Examples | Retention Period |
|---|---|---|
| Account data | Name, email, password hash, membership tier | Life of account + 3 years |
| Order data | Items, amounts, dates, delivery address | 7 years (tax requirement) |
| Payment data | Last four digits, card brand, expiry; **never** full card numbers | 7 years |
| Support interactions | Chat transcripts, call recordings, email threads | 24 months |
| Device and usage data | IP address, browser, pages viewed | 13 months |
| Marketing preferences | Subscription status, consent timestamps | Until withdrawn + 3 years |
| Trade-in data | Device serial, condition grade, appraisal | 3 years |

Full payment card numbers are never stored on ShopSmart systems. Card data is tokenized by the payment processor at entry; ShopSmart retains only the token, the last four digits and the card brand.

### 2.2 How Data Is Used

Order and account data are used to fulfil orders, process returns and refunds, provide support, detect fraud, meet tax and accounting obligations, and — where the customer has consented — send marketing communications. Support transcripts are used for quality assurance and agent training.

---

## 3. Customer Data Rights

### 3.1 Available Rights

| Right | How to Exercise | Response Time |
|---|---|---|
| Access a copy of your data | Account Settings → Privacy → Request My Data | 30 days |
| Correct inaccurate data | Account Settings, or contact support | 7 days |
| Delete your account and data | Account Settings → Close Account | 30 days |
| Withdraw marketing consent | Unsubscribe link, or Account Settings | Immediate |
| Object to automated decisions | Contact the privacy team | 30 days |
| Port data to another service | Request My Data (machine-readable export) | 30 days |

### 3.2 Limits on Deletion

Account closure deletes marketing preferences, device and usage data, and support transcripts older than the minimum retention period. It does **not** delete order and payment records, which are retained for 7 years to satisfy tax and accounting obligations. Retained order records are restricted to finance and compliance access and are no longer used for marketing or personalization.

Deletion cannot be completed while an order is in progress, a return is open, a warranty claim is active, or a payment dispute is unresolved. In those cases the request is queued and processed once the matter closes.

---

## 4. Data Sharing

### 4.1 Who Receives Data

| Recipient | Data Shared | Purpose |
|---|---|---|
| Shipping carriers | Name, address, phone, order ID | Delivery |
| Payment processors | Tokenized payment data, amount | Payment and refunds |
| BNPL providers | Name, email, amount, order ID | Instalment plan administration |
| Manufacturers | Serial number, proof of purchase | Warranty claims, at customer request |
| Fraud screening provider | Order, device and payment signals | Risk assessment |
| Tax authorities | Transaction records | Legal obligation |

ShopSmart does not sell customer personal information. Data is never shared with third-party advertisers in a form that identifies an individual.

### 4.2 International Transfers

Data may be processed in countries other than the customer's own, under contractual safeguards requiring equivalent protection. Customers may request details of the safeguards in place by contacting the privacy team.

---

## 5. Security Measures

### 5.1 Technical Controls

- All data in transit protected by TLS 1.2 or higher
- Data at rest encrypted using AES-256
- Passwords stored as salted hashes; plaintext passwords are never stored or logged
- Multi-factor authentication available on all accounts and mandatory for Business accounts
- Payment card data tokenized at entry; ShopSmart systems are never in scope for full card data
- Access to production customer data restricted by role and logged for audit

### 5.2 What ShopSmart Will Never Ask For

ShopSmart staff and automated systems will never ask a customer for their full payment card number, their account password, a one-time authentication code, or remote access to their device. Any communication making such a request is fraudulent and should be reported to security@shopsmartelectronics.com.

### 5.3 Breach Notification

In the event of a data breach affecting personal data, ShopSmart notifies affected customers and the relevant supervisory authority within **72 hours** of becoming aware, describing the data involved, the likely consequences and the steps being taken.

---

## 6. Automated Decision-Making

### 6.1 Where Automation Is Used

Automated systems are used in fraud screening, return-abuse detection, credit eligibility for financing, and in the AI support assistant that answers policy questions and retrieves order details.

### 6.2 Safeguards

Automated decisions that materially affect a customer — declining an order for fraud risk, suspending an account for return abuse, or refusing financing — are reviewable by a human on request. Customers may contact support to request human review, and the reviewing agent has authority to overturn the automated decision.

The AI support assistant is restricted to information the authenticated customer is entitled to see. It must decline requests for another customer's records, must not state policy terms it cannot support with a retrieved source document, and must escalate to a human agent when the customer disputes its determination or when the request falls outside its documented capabilities.

---

## 7. Contact

- **Privacy Team:** privacy@shopsmartelectronics.com
- **Security Reports:** security@shopsmartelectronics.com
- **Data Protection Officer:** dpo@shopsmartelectronics.com
- **Phone:** 1-800-746-7762, option 9
- **Postal:** ShopSmart Electronics, Attn: Privacy, 1200 Commerce Way, Springfield, IL 62701

---

*This policy is subject to change. Material changes are notified by email at least 30 days before taking effect.*
