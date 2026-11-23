//+------------------------------------------------------------------+
//| ExportM1.mq5 (regime-vol-forecast): READ-ONLY M1 bar export.      |
//| No trading calls.                                                 |
//|                                                                   |
//| For each symbol in <DIR>symbols.txt:                              |
//|  - waits until the terminal has loaded the full M1 history,       |
//|  - writes all M1 bars to <DIR><SYM>_M1.csv in the MT5 export      |
//|    format (tab-separated, CRLF, header <DATE>..<SPREAD>),         |
//|  - writes symbol facts (chart mode, digits, sessions, history     |
//|    depth, bars per year) to <DIR>export_meta.tsv.                 |
//| Symbols that were not in Market Watch are removed again.          |
//+------------------------------------------------------------------+
const string   DIR        = "rvf_export\\";
const datetime FROM       = D'2000.01.01 00:00';
const datetime TO         = D'2030.01.01 00:00';
const int      FIRST_YEAR = 2000;
const int      LAST_YEAR  = 2029;
const int      MAX_WAIT_S = 900;      // per symbol, for the history download

void Put(int h, string &s) { uchar a[]; int n = StringToCharArray(s, a, 0, WHOLE_ARRAY, CP_ACP); FileWriteArray(h, a, 0, n - 1); s = ""; }
void W(int h, string s) { string x = s + "\r\n"; Put(h, x); }
void Meta(int h, string sym, string key, string val) { W(h, sym + "\t" + key + "\t" + val); FileFlush(h); }
string HM(datetime t) { long s = (long)t; return StringFormat("%02d:%02d", (int)(s / 3600), (int)((s % 3600) / 60)); }

// Broker-declared sessions per weekday (0 = Sunday), e.g. "1:01:00-23:15,23:30-24:00"
string Sessions(string sym, bool quote)
  {
   string out = "";
   for(int d = 0; d < 7; d++)
     {
      string day = "";
      for(uint i = 0; i < 12; i++)
        {
         datetime a, b;
         bool ok = quote ? SymbolInfoSessionQuote(sym, (ENUM_DAY_OF_WEEK)d, i, a, b)
                         : SymbolInfoSessionTrade(sym, (ENUM_DAY_OF_WEEK)d, i, a, b);
         if(!ok) break;
         day += (i > 0 ? "," : "") + HM(a) + "-" + HM(b);
        }
      out += (d > 0 ? " | " : "") + IntegerToString(d) + ":" + day;
     }
   return out;
  }

// Polls until the M1 series is synchronised, reaches the server's first date and the bar count is stable.
long WaitHistory(string sym, int h)
  {
   long n = -1, prev = -2; int stable = 0; bool sync = false, deep = false;
   datetime first = 0, sfirst = 0;
   uint t0 = GetTickCount();
   MqlRates tmp[];
   // offline nothing can be downloaded and every request blocks for seconds: one short pass over the local cache only
   uint limit = TerminalInfoInteger(TERMINAL_CONNECTED) ? (uint)MAX_WAIT_S : 20;
   while((GetTickCount() - t0) / 1000 < limit)
     {
      CopyRates(sym, PERIOD_M1, FROM, FROM + 30 * 86400, tmp);        // asking for the oldest range triggers the deep download
      n      = Bars(sym, PERIOD_M1, FROM, TO);
      sync   = (bool)SeriesInfoInteger(sym, PERIOD_M1, SERIES_SYNCHRONIZED);
      first  = (datetime)SeriesInfoInteger(sym, PERIOD_M1, SERIES_FIRSTDATE);
      sfirst = (datetime)SeriesInfoInteger(sym, PERIOD_M1, SERIES_SERVER_FIRSTDATE);
      stable = (n == prev) ? stable + 1 : 0;
      deep   = (sfirst > 0 && first > 0 && first <= sfirst + 7 * 86400);
      if(n > 0 && sync && deep && stable >= 6) break;                  // complete: 3 s without change
      if(stable >= 120) break;                                         // 60 s without change: offline or nothing more to load
      prev = n; Sleep(500);
     }
   Meta(h, sym, "wait_seconds", DoubleToString((GetTickCount() - t0) / 1000.0, 1));
   Meta(h, sym, "series_synchronized", sync ? "1" : "0");
   Meta(h, sym, "terminal_first_date", TimeToString(first, TIME_DATE | TIME_MINUTES));
   Meta(h, sym, "server_first_date", TimeToString(sfirst, TIME_DATE | TIME_MINUTES));
   Meta(h, sym, "reached_server_first_date", deep ? "1" : "0");
   Meta(h, sym, "bars_reported", IntegerToString(n));
   return n;
  }

void Export(string sym, int meta)
  {
   uint t0 = GetTickCount();
   bool was = (bool)SymbolInfoInteger(sym, SYMBOL_SELECT);
   if(!was && !SymbolSelect(sym, true)) { Meta(meta, sym, "status", "NOT_AVAILABLE"); return; }
   int dg = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   ENUM_SYMBOL_CHART_MODE cm = (ENUM_SYMBOL_CHART_MODE)SymbolInfoInteger(sym, SYMBOL_CHART_MODE);
   Meta(meta, sym, "description", SymbolInfoString(sym, SYMBOL_DESCRIPTION));
   Meta(meta, sym, "path", SymbolInfoString(sym, SYMBOL_PATH));
   Meta(meta, sym, "chart_mode", cm == SYMBOL_CHART_MODE_BID ? "bid" : (cm == SYMBOL_CHART_MODE_LAST ? "last" : "unknown"));
   Meta(meta, sym, "digits", IntegerToString(dg));
   Meta(meta, sym, "point", DoubleToString(SymbolInfoDouble(sym, SYMBOL_POINT), 8));
   Meta(meta, sym, "spread_float", IntegerToString(SymbolInfoInteger(sym, SYMBOL_SPREAD_FLOAT)));
   Meta(meta, sym, "calc_mode", IntegerToString(SymbolInfoInteger(sym, SYMBOL_TRADE_CALC_MODE)));
   Meta(meta, sym, "currency_profit", SymbolInfoString(sym, SYMBOL_CURRENCY_PROFIT));
   Meta(meta, sym, "sessions_quote", Sessions(sym, true));
   Meta(meta, sym, "sessions_trade", Sessions(sym, false));

   long reported = WaitHistory(sym, meta);
   if(reported <= 0) { Meta(meta, sym, "status", "NO_BARS"); if(!was) SymbolSelect(sym, false); return; }
   // only the years the terminal has: a request for years without data blocks for a long time when offline
   MqlDateTime d0, d1;
   datetime have_from = (datetime)SeriesInfoInteger(sym, PERIOD_M1, SERIES_FIRSTDATE);
   TimeToStruct(have_from > 0 ? have_from : FROM, d0);
   TimeToStruct(TimeLocal() + 2 * 86400, d1);

   int h = FileOpen(DIR + sym + "_M1.csv", FILE_WRITE | FILE_BIN);
   if(h == INVALID_HANDLE) { Meta(meta, sym, "status", "FILE_ERROR"); if(!was) SymbolSelect(sym, false); return; }
   string buf = "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\r\n";
   long total = 0; int errors = 0; datetime t_first = 0, t_last = 0; string per_year = "";
   for(int y = MathMax(FIRST_YEAR, d0.year); y <= MathMin(LAST_YEAR, d1.year); y++)
     {
      datetime a = StringToTime(StringFormat("%04d.01.01 00:00", y));
      datetime b = StringToTime(StringFormat("%04d.01.01 00:00", y + 1)) - 60;
      if(Bars(sym, PERIOD_M1, a, b) <= 0) continue;
      MqlRates r[];
      ResetLastError();
      int n = CopyRates(sym, PERIOD_M1, a, b, r);
      if(n < 0) { errors++; per_year += IntegerToString(y) + ":ERR" + IntegerToString(GetLastError()) + " "; continue; }
      if(n == 0) continue;
      for(int j = 0; j < n; j++)
        {
         StringAdd(buf, TimeToString(r[j].time, TIME_DATE) + "\t" + TimeToString(r[j].time, TIME_SECONDS) + "\t" +
                   DoubleToString(r[j].open, dg) + "\t" + DoubleToString(r[j].high, dg) + "\t" + DoubleToString(r[j].low, dg) + "\t" +
                   DoubleToString(r[j].close, dg) + "\t" + IntegerToString(r[j].tick_volume) + "\t" + IntegerToString(r[j].real_volume) + "\t" +
                   IntegerToString(r[j].spread) + "\r\n");
         if(j % 5000 == 4999) Put(h, buf);
        }
      if(t_first == 0) t_first = r[0].time;
      t_last = r[n - 1].time;
      total += n;
      per_year += IntegerToString(y) + ":" + IntegerToString(n) + " ";
     }
   if(StringLen(buf) > 0) Put(h, buf);
   FileClose(h);
   if(!was) SymbolSelect(sym, false);
   Meta(meta, sym, "bars_written", IntegerToString(total));
   Meta(meta, sym, "bars_per_year", per_year);
   Meta(meta, sym, "first_bar", TimeToString(t_first, TIME_DATE | TIME_MINUTES));
   Meta(meta, sym, "last_bar", TimeToString(t_last, TIME_DATE | TIME_MINUTES));
   Meta(meta, sym, "copy_errors", IntegerToString(errors));
   Meta(meta, sym, "seconds", DoubleToString((GetTickCount() - t0) / 1000.0, 1));
   Meta(meta, sym, "status", (total > 0 && errors == 0 && total == reported) ? "OK" : (total > 0 ? "CHECK" : "NO_BARS"));
  }

void OnStart()
  {
   int fin = FileOpen(DIR + "symbols.txt", FILE_READ | FILE_TXT | FILE_ANSI);
   if(fin == INVALID_HANDLE) return;
   string syms[]; int ns = 0;
   while(!FileIsEnding(fin)) { string s = FileReadString(fin); StringTrimRight(s); if(StringLen(s) > 0) { ArrayResize(syms, ns + 1); syms[ns++] = s; } }
   FileClose(fin);

   int meta = FileOpen(DIR + "export_meta.tsv", FILE_WRITE | FILE_BIN);
   W(meta, "symbol\tkey\tvalue");
   // give the terminal up to 30 s to connect before the first request
   for(int k = 0; k < 60 && !TerminalInfoInteger(TERMINAL_CONNECTED); k++) Sleep(500);
   Meta(meta, "_terminal", "connected", IntegerToString(TerminalInfoInteger(TERMINAL_CONNECTED)));
   Meta(meta, "_terminal", "build", IntegerToString(TerminalInfoInteger(TERMINAL_BUILD)));
   Meta(meta, "_terminal", "max_bars", IntegerToString(TerminalInfoInteger(TERMINAL_MAXBARS)));
   Meta(meta, "_terminal", "account_server", AccountInfoString(ACCOUNT_SERVER));
   Meta(meta, "_terminal", "account_company", AccountInfoString(ACCOUNT_COMPANY));
   Meta(meta, "_terminal", "time_trade_server", TimeToString(TimeTradeServer(), TIME_DATE | TIME_SECONDS));
   Meta(meta, "_terminal", "time_gmt", TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS));
   Meta(meta, "_terminal", "server_minus_gmt_hours", DoubleToString((double)(TimeTradeServer() - TimeGMT()) / 3600.0, 2));

   // offline runs export only the local cache: allowed only when the runner asks for it (offline_ok.txt)
   if(!TerminalInfoInteger(TERMINAL_CONNECTED) && !FileIsExist(DIR + "offline_ok.txt"))
     {
      Meta(meta, "_terminal", "aborted", "not connected");
      FileClose(meta);
      return;
     }

   for(int i = 0; i < ns; i++) Export(syms[i], meta);

   Meta(meta, "_terminal", "connected_at_end", IntegerToString(TerminalInfoInteger(TERMINAL_CONNECTED)));
   Meta(meta, "_terminal", "done", IntegerToString(ns));
   FileClose(meta);
   Print("regime-vol-forecast M1 export done: ", ns);
  }
