'use client';

import { useEffect, useState, useRef } from 'react';
import { useParams } from 'next/navigation';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, Legend,
  AreaChart, Area, XAxis, YAxis, CartesianGrid 
} from 'recharts';
import { toPng } from 'html-to-image';
import jsPDF from 'jspdf';
import Link from 'next/link';

export default function PortfolioDetailPage() {
  const params = useParams();
  const id = params?.id;
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [portfolioData, setPortfolioData] = useState<any>(null);
  
  const [allocationData, setAllocationData] = useState<any[]>([]);
  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [isDownloading, setIsDownloading] = useState(false);
  const dashboardRef = useRef<HTMLDivElement>(null);

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#14b8a6', '#f43f5e', '#f97316'];

  useEffect(() => {
    if (!id) return;
    
    async function fetchPortfolio() {
      setIsLoading(true);
      try {
        const res = await fetch(`http://localhost:8000/api/portfolios/${id}`);
        if (!res.ok) throw new Error('ไม่สามารถโหลดข้อมูลพอร์ตการลงทุนได้');
        const data = await res.json();
        
        if (data.status === 'success') {
          setPortfolioData(data);
          
          // 1. จัดการข้อมูลสัดส่วนหุ้น (วงกลม)
          if (data.portfolio && Array.isArray(data.portfolio)) {
            const mappedAllocation = data.portfolio.map((stock: any, index: number) => ({
              name: stock.Ticker.replace('.BK', ''), 
              value: Number((stock.Weight * 100).toFixed(2)), 
              color: COLORS[index % COLORS.length]
            }));
            setAllocationData(mappedAllocation);
          }
          
          // 2. จัดการข้อมูล Backtest
          if (data.backtest && data.backtest.chart_data && Array.isArray(data.backtest.chart_data)) {
            const initialBudget = data.metadata.budget || 100000;
            const ratio = initialBudget / 100000;
            const mappedPerformance = data.backtest.chart_data.map((bt: any) => ({
              year: bt.date, 
              AI: Math.round(bt.AI * ratio),        
              SET50: Math.round(bt.SET50 * ratio)   
            }));
            setPerformanceData(mappedPerformance);
          }
        } else {
          setError(data.message || 'เกิดข้อผิดพลาดในการโหลดข้อมูล');
        }
      } catch (err: any) {
        setError(err.message || 'เกิดข้อผิดพลาดในการเชื่อมต่อเซิร์ฟเวอร์');
      } finally {
        setIsLoading(false);
      }
    }
    
    fetchPortfolio();
  }, [id]);

  const handleDownloadPDF = async () => {
    if (!dashboardRef.current) return;
    setIsDownloading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      const targetWidth = 1600;
      const dataUrl = await toPng(dashboardRef.current, {
        cacheBust: true, backgroundColor: '#ffffff', quality: 1.0, pixelRatio: 2,
        width: targetWidth, style: { width: `${targetWidth}px`, maxWidth: `${targetWidth}px`, height: 'auto', margin: '0', padding: '40px' }
      });
      const img = new Image();
      img.src = dataUrl;
      await new Promise((resolve) => { img.onload = resolve; });
      const pdf = new jsPDF('l', 'px', [img.width, img.height]);
      pdf.addImage(dataUrl, 'PNG', 0, 0, img.width, img.height);
      pdf.save(`IntelliPort_Report_${id}.pdf`);
    } catch (error: any) {
      console.error("PDF Error:", error);
      alert(`บันทึกไม่สำเร็จ: {error.message}`);
    } finally {
      setIsDownloading(false);
    }
  };

  const RADIAN = Math.PI / 180;
  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
    if (percent < 0.03) return null;
    return (
      <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" className="text-xs font-bold shadow-sm">
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#FFFEF5]">
        <div className="text-center">
          <svg className="animate-spin h-10 w-10 text-yellow-500 mx-auto mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <p className="text-slate-600 font-medium">กำลังโหลดข้อมูลและคำนวณ Backtest ย้อนหลัง...</p>
        </div>
      </div>
    );
  }

  if (error || !portfolioData) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#FFFEF5] px-4">
        <div className="max-w-md w-full bg-white p-8 rounded-2xl shadow-xl border border-red-100 text-center">
          <span className="text-4xl">⚠️</span>
          <h2 className="text-xl font-bold text-slate-800 mt-4 mb-2">เกิดข้อผิดพลาด</h2>
          <p className="text-slate-600 mb-6">{error || 'ไม่พบข้อมูลพอร์ตการลงทุนนี้'}</p>
          <Link href="/dashboard" className="inline-block bg-slate-900 hover:bg-slate-800 text-white font-bold py-3 px-6 rounded-xl transition-all">
            กลับไปหน้าแดชบอร์ด
          </Link>
        </div>
      </div>
    );
  }

  const meta = portfolioData.metadata;
  let riskLevel = 'Moderate';
  let riskBadgeColor = 'bg-yellow-100 text-yellow-700';
  if (meta.target_beta < 0.9) {
    riskLevel = 'Conservative';
    riskBadgeColor = 'bg-green-100 text-green-700';
  } else if (meta.target_beta > 1.1) {
    riskLevel = 'Aggressive';
    riskBadgeColor = 'bg-red-100 text-red-700';
  }

  return (
    <main className="min-h-screen bg-[#FFFEF5] py-8 px-4 sm:px-6 lg:px-8 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] selection:bg-yellow-400 selection:text-black">
      <div className="max-w-7xl mx-auto mb-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <div className="flex items-center gap-3">
            <Link href="/dashboard" className="text-slate-500 hover:text-slate-800 transition-colors flex items-center gap-1 font-semibold">
              <span>⬅️</span> ย้อนกลับไปประวัติพอร์ต
            </Link>
          </div>
          <h1 className="text-3xl font-bold text-slate-900 mt-2">{meta.name}</h1>
          <p className="text-slate-500 mt-1">รายละเอียดพอร์ตการลงทุน (พอร์ต ID: #{id})</p>
        </div>
        <button 
          onClick={handleDownloadPDF} disabled={isDownloading}
          className={`flex items-center px-6 py-3 border rounded-xl shadow-sm transition-all font-medium ${
            isDownloading ? 'bg-blue-50 border-blue-200 text-blue-600 cursor-wait' : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50 hover:shadow-md'
          }`}
        >
          {isDownloading ? 'กำลังสร้างไฟล์...' : '⬇️ ดาวน์โหลด PDF'}
        </button>
      </div>

      <div className="max-w-7xl mx-auto bg-white p-8 rounded-3xl shadow-sm border border-slate-200" ref={dashboardRef}> 
        <div className="mb-8 border-b border-slate-100 pb-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div>
            <h2 className="text-2xl font-bold text-slate-900">Portfolio Details</h2>
            <p className="text-slate-500">ผลการวิเคราะห์พอร์ตการลงทุนด้วยระบบสมองกล AI</p>
          </div>
          <div className="text-slate-400 text-xs font-mono">
            วันที่วิเคราะห์: {new Date(meta.created_at).toLocaleString('th-TH')}
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3 mb-8">
           <div className="bg-slate-50 p-4 rounded-2xl border border-slate-100">
              <span className="text-slate-500 text-xs block mb-1 font-semibold">เงินลงทุนตั้งต้น</span>
              <span className="text-lg sm:text-xl font-bold text-slate-800">฿{Number(meta.budget).toLocaleString()}</span>
           </div>
           <div className="bg-slate-50 p-4 rounded-2xl border border-slate-100">
              <span className="text-slate-500 text-xs block mb-1 font-semibold">เป้าหมายเงินออม</span>
              <span className="text-lg sm:text-xl font-bold text-indigo-600">
                {meta.target_amount ? `฿${Number(meta.target_amount).toLocaleString()}` : '-'}
              </span>
           </div>
           <div className="bg-slate-50 p-4 rounded-2xl border border-slate-100">
              <span className="text-slate-500 text-xs block mb-1 font-semibold">คาดการณ์เมื่อครบ {meta.duration_years} ปี</span>
              {meta.forecast_lower !== undefined && meta.forecast_lower !== null && meta.forecast_upper !== undefined && meta.forecast_upper !== null ? (
                <span className="text-[11px] sm:text-[13px] font-bold text-slate-800 block mt-1.5">
                  ฿{Math.round(meta.forecast_lower).toLocaleString()} ~ ฿{Math.round(meta.forecast_upper).toLocaleString()}
                </span>
              ) : (
                <span className="text-lg sm:text-xl font-bold text-slate-400 block mt-1">-</span>
              )}
           </div>
           <div className="bg-slate-50 p-4 rounded-2xl border border-slate-100">
              <span className="text-slate-500 text-xs block mb-1 font-semibold">โอกาสสำเร็จ</span>
              {meta.success_probability !== undefined && meta.success_probability !== null ? (
                <span className={`text-lg sm:text-xl font-bold ${meta.success_probability >= 0.7 ? 'text-green-600' : meta.success_probability >= 0.4 ? 'text-amber-500' : 'text-red-500'}`}>
                  {(meta.success_probability * 100).toFixed(0)}%
                </span>
              ) : (
                <span className="text-lg sm:text-xl font-bold text-slate-400">-</span>
              )}
           </div>
           <div className="bg-slate-50 p-4 rounded-2xl border border-slate-100">
              <span className="text-slate-500 text-xs block mb-1 font-semibold">ผลตอบแทนคาดหวังรายปี</span>
              <span className="text-lg sm:text-xl font-bold text-green-600">{(meta.expected_return * 100).toFixed(2)}%</span>
           </div>
           <div className="bg-slate-50 p-4 rounded-2xl border border-slate-100 flex justify-between items-center">
              <div>
                <span className="text-slate-500 text-xs block mb-1 font-semibold">ระดับความเสี่ยง</span>
                <span className="text-base sm:text-lg font-bold text-slate-800">{riskLevel}</span>
              </div>
              <span className={`px-2 py-0.5 rounded-full text-[10px] sm:text-xs font-bold ${riskBadgeColor} whitespace-nowrap`}>
                Beta {meta.target_beta}
              </span>
           </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1 p-6 rounded-2xl border border-slate-100 bg-white shadow-sm hover:shadow-md transition-shadow">
            <h3 className="text-lg font-bold text-slate-900 mb-6 border-l-4 border-blue-500 pl-3">
              สัดส่วนสินทรัพย์แนะนำ
            </h3>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={allocationData} innerRadius={60} outerRadius={100} paddingAngle={5} dataKey="value"
                    labelLine={false} label={renderCustomizedLabel}
                  >
                    {allocationData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <RechartsTooltip
                    formatter={(value: any) => {
                      const pct = Number(value);
                      const initialBudget = meta.budget || 100000;
                      const amount = (pct / 100) * initialBudget;
                      return [`${pct}% (฿${Math.round(amount).toLocaleString()})`];
                    }}
                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                  />
                  <Legend verticalAlign="bottom" height={72} align="center" iconType="circle" wrapperStyle={{ paddingTop: '20px' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="lg:col-span-2 p-6 rounded-2xl border border-slate-100 bg-white shadow-sm hover:shadow-md transition-shadow">
            <h3 className="text-lg font-bold text-slate-900 mb-6 border-l-4 border-green-500 pl-3">
              ผลการทดสอบย้อนหลัง (Backtest: {meta.duration_years} ปี)
            </h3>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={performanceData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorAI" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="year" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `฿${(value/1000).toFixed(0)}k`} width={60} />
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <Area type="monotone" dataKey="SET50" stroke="#cbd5e1" strokeWidth={3} fill="transparent" name="ดัชนีตลาด SET50" />
                  <Area type="monotone" dataKey="AI" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorAI)" name="พอร์ต AI แนะนำ" />
                  <RechartsTooltip contentStyle={{ backgroundColor: '#fff', borderRadius: '10px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                  <Legend verticalAlign="top" height={36}/>
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
