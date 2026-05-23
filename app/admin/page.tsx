'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';

export const dynamic = "force-dynamic";

export default function AdminPage() {
  const { userId } = useAuth();
  const router = useRouter();

  const [isAdminAuthenticated, setIsAdminAuthenticated] = useState(false);
  const [adminPassword, setAdminPassword] = useState('');
  const [passwordError, setPasswordError] = useState('');

  const [activeTab, setActiveTab] = useState<'users' | 'assets'>('users');
  const [users, setUsers] = useState<any[]>([]);
  const [assets, setAssets] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const auth = localStorage.getItem('isAdmin') === 'true';
    setIsAdminAuthenticated(auth);
    if (auth) {
      fetchAdminData();
    }
  }, []);

  const handleAdminLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (adminPassword === 'admin') {
      localStorage.setItem('isAdmin', 'true');
      setIsAdminAuthenticated(true);
      setPasswordError('');
      fetchAdminData();
      // Dispatch event to update navbar immediately
      window.dispatchEvent(new Event('admin-login'));
    } else {
      setPasswordError('รหัสผ่านไม่ถูกต้อง กรุณาลองใหม่อีกครั้ง');
    }
  };

  const handleExitAdmin = () => {
    localStorage.removeItem('isAdmin');
    setIsAdminAuthenticated(false);
    setAdminPassword('');
    window.dispatchEvent(new Event('admin-logout'));
    router.push('/dashboard');
  };

  const fetchAdminData = async () => {
    setIsLoading(true);
    try {
      const [usersRes, assetsRes] = await Promise.all([
        fetch('http://localhost:8000/api/admin/users'),
        fetch('http://localhost:8000/api/admin/assets')
      ]);

      const usersData = await usersRes.json();
      const assetsData = await assetsRes.json();

      if (usersData.status === 'success') {
        setUsers(usersData.data);
      }
      if (assetsData.status === 'success') {
        setAssets(assetsData.data);
      }
    } catch (err) {
      console.error('Failed to fetch admin data', err);
    } finally {
      setIsLoading(false);
    }
  };



  if (!isAdminAuthenticated) {
    return (
      <main className="min-h-screen bg-[#FFFEF5] flex items-center justify-center py-20 px-4 sm:px-6 lg:px-8 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] font-sans">
        <div className="max-w-md w-full p-8 bg-white/80 backdrop-blur-2xl rounded-2xl shadow-2xl border border-gray-100/50">
          <div className="flex flex-col items-center mb-6">
            <span className="text-4xl mb-3">🔑</span>
            <h2 className="text-2xl font-bold text-slate-900">เข้าสู่ระบบผู้ดูแลระบบ (Admin)</h2>
            <p className="text-slate-500 text-sm mt-1">กรุณากรอกรหัสผ่านผู้ดูแลระบบเพื่อเข้าถึงข้อมูล</p>
          </div>
          <form onSubmit={handleAdminLogin} className="space-y-4">
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-1">รหัสผ่าน Admin</label>
              <input
                type="password"
                value={adminPassword}
                onChange={(e) => setAdminPassword(e.target.value)}
                className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:border-transparent transition-all text-slate-800"
                placeholder="ป้อนรหัสผ่าน..."
                required
              />
            </div>
            {passwordError && (
              <p className="text-red-500 text-xs font-semibold">{passwordError}</p>
            )}
            <button
              type="submit"
              className="w-full py-3 bg-yellow-400 hover:bg-yellow-500 text-blue-950 font-bold rounded-xl transition-all shadow-md shadow-yellow-400/20"
            >
              ยืนยันรหัสผ่าน
            </button>
          </form>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-[#FFFEF5] py-12 px-4 sm:px-6 lg:px-8 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')]">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-end mb-8">
          <div>
            <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight">ระบบผู้ดูแลระบบ</h1>
            <p className="mt-2 text-lg text-slate-600">จัดการข้อมูลผู้ใช้งานและตั้งค่าพารามิเตอร์สินทรัพย์</p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={handleExitAdmin}
              className="bg-red-50 hover:bg-red-100 text-red-600 font-bold px-4 py-2 rounded-lg text-sm transition-all border border-red-200"
            >
              ออกจากโหมด Admin
            </button>
            <div className="bg-yellow-100 text-yellow-800 px-4 py-2 rounded-lg font-bold shadow-sm">
              Admin Mode
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex space-x-2 border-b border-slate-200 mb-8">
          <button
            onClick={() => setActiveTab('users')}
            className={`py-3 px-6 font-bold text-lg border-b-4 transition-colors ${activeTab === 'users' ? 'border-yellow-400 text-yellow-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
          >
            👥 ข้อมูลผู้ใช้งาน ({users.length})
          </button>
          <button
            onClick={() => setActiveTab('assets')}
            className={`py-3 px-6 font-bold text-lg border-b-4 transition-colors ${activeTab === 'assets' ? 'border-yellow-400 text-yellow-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
          >
            📊 สินทรัพย์ SET50 ({assets.length})
          </button>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-20">
            <svg className="animate-spin h-10 w-10 text-yellow-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </div>
        ) : (
          <div className="bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden">
            {activeTab === 'users' && (
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-100 text-slate-600 text-sm">
                      <th className="p-4 font-semibold">Clerk User ID</th>
                      <th className="p-4 font-semibold">สร้างบัญชีเมื่อ</th>
                      <th className="p-4 font-semibold">ล็อกอินล่าสุด</th>
                      <th className="p-4 font-semibold text-center">จำนวนแผนการลงทุน</th>
                    </tr>
                  </thead>
                  <tbody>
                    {users.length > 0 ? users.map((u, i) => (
                      <tr key={i} className="border-b border-slate-50 hover:bg-slate-50 transition-colors">
                        <td className="p-4 text-slate-800 font-mono text-sm">{u.clerk_id}</td>
                        <td className="p-4 text-slate-600">{new Date(u.created_at).toLocaleString('th-TH')}</td>
                        <td className="p-4 text-slate-600">{new Date(u.last_login_at).toLocaleString('th-TH')}</td>
                        <td className="p-4 text-center">
                          <span className="inline-flex items-center justify-center px-3 py-1 rounded-full bg-yellow-100 text-yellow-800 font-bold">
                            {u.portfolio_count}
                          </span>
                        </td>
                      </tr>
                    )) : (
                      <tr><td colSpan={4} className="p-8 text-center text-slate-500">ไม่พบข้อมูลผู้ใช้งาน</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}

            {activeTab === 'assets' && (
              <div className="p-6">
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                  {assets.map((ticker, i) => (
                    <div key={i} className="flex items-center justify-between p-3 bg-slate-50 border border-slate-200 rounded-lg hover:border-yellow-300 transition-colors">
                      <span className="font-bold text-slate-800">{ticker}</span>
                      <span className="text-xs text-green-600 font-semibold bg-green-100 px-2 py-1 rounded">Active</span>
                    </div>
                  ))}
                </div>
                <p className="mt-6 text-sm text-slate-500">* ข้อมูลดึงมาจาก SET50 ล่าสุดโดยอัตโนมัติ น้ำหนักจำกัดสูงสุดถูกควบคุมไว้ใน Backend ไม่เกิน 25% ต่อตัว</p>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
