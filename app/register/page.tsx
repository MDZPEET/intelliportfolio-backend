'use client';

import React from 'react';
import { SignUp } from "@clerk/nextjs";
import Link from 'next/link';

export default function RegisterPage() {
  return (
    <div className="relative min-h-screen flex items-center justify-center bg-white overflow-hidden font-sans">
      
      {/* 1. Background Decor เดิมของคุณ (รักษาความสวยงามตาม Section 10.2.11) */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#f1f5f9_1px,transparent_1px),linear-gradient(to_bottom,#f1f5f9_1px,transparent_1px)] bg-[size:40px_40px]"></div>
      </div>
      
      {/* 2. Register Card ที่ใช้ Clerk SignUp */}
      <div className="relative z-10 w-full max-w-md bg-white/80 backdrop-blur-xl p-8 rounded-3xl shadow-2xl border border-slate-100">
        
        {/* ส่วนหัว Logo เดิมของคุณ */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-12 h-12 bg-blue-600 rounded-xl mb-4 shadow-lg shadow-blue-500/30">
            <span className="text-white font-bold text-xl">IP</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-900 mb-2">สร้างบัญชีผู้ใช้ใหม่</h1>
          <p className="text-slate-500 text-sm">เพื่อเข้าถึงระบบจัดการพอร์ตอัจฉริยะ</p>
        </div>

        {/* 3. ระบบสมัครสมาชิกของ Clerk */}
        <SignUp 
          appearance={{
            elements: {
              formButtonPrimary: "bg-yellow-400 hover:bg-yellow-500 text-slate-900 font-bold py-3 rounded-xl shadow-lg shadow-yellow-400/30 transition-all",
              card: "shadow-none bg-transparent border-none p-0",
              headerTitle: "hidden", 
              headerSubtitle: "hidden",
              socialButtonsBlockButton: "rounded-xl border-slate-200",
              formFieldInput: "rounded-xl border-slate-200 bg-white",
              footerActionLink: "text-blue-600 font-bold hover:text-blue-800"
            }
          }}
          routing="path"
          path="/register"
          signInUrl="/login"
          forceRedirectUrl="/dashboard" // เมื่อสมัครเสร็จให้ไปหน้า Dashboard ทันที
        />

        <div className="mt-6 pt-4 border-t border-slate-100 text-center flex flex-col items-center gap-1">
           <span className="text-[10px] uppercase tracking-widest text-slate-400 font-bold">
             Project CS-01: IntelliPort
           </span>
           <Link href="/admin" className="text-[11px] text-slate-400 hover:text-blue-600 hover:underline transition-colors mt-1.5">
             สำหรับผู้ดูแลระบบ (Admin)
           </Link>
        </div>
      </div>
    </div>
  );
}