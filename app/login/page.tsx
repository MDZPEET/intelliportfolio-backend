'use client';

import React from 'react';
import { SignIn } from "@clerk/nextjs";
import Link from 'next/link';

export default function LoginPage() {
  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-white relative overflow-hidden font-sans">
      
      {/* 1. เก็บ Background Decor เดิมของคุณไว้ทั้งหมด */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
      <div className="absolute top-20 left-20 w-72 h-72 bg-yellow-100 rounded-full mix-blend-multiply filter blur-2xl opacity-40 animate-pulse"></div>
      <div className="absolute bottom-20 right-20 w-72 h-72 bg-blue-100 rounded-full mix-blend-multiply filter blur-2xl opacity-40 animate-pulse delay-700"></div>

      {/* 2. Login Card ที่ปรับแต่งใหม่ */}
      <div className="relative z-10 w-full max-w-md p-8 bg-white/80 backdrop-blur-2xl rounded-2xl shadow-2xl border border-gray-100/50">
        
        {/* ส่วนหัว Logo เดิมของคุณ */}
        <div className="flex flex-col items-center mb-8">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center text-white font-bold text-2xl shadow-lg shadow-blue-600/20 transform -rotate-3 transition-transform cursor-default">
              IP
            </div>
            <span className="text-3xl font-bold text-blue-950 tracking-tight">IntelliPort</span>
          </div>
          <p className="text-gray-500 text-sm font-medium italic">Smart Portfolio Management System</p>
        </div>

        {/* 3. แทนที่ Form เดิมด้วย Clerk SignIn Component */}
        {/* เราสามารถปรับแต่งสีปุ่มให้เหลืองเหมือนเดิมได้ผ่าน appearance prop */}
        <SignIn 
          appearance={{
            elements: {
              formButtonPrimary: "bg-yellow-400 hover:bg-yellow-500 text-blue-950 font-bold",
              card: "shadow-none bg-transparent border-none p-0", // ลบเงาซ้ำซ้อนออก
              headerTitle: "hidden", // ซ่อนหัวข้อซ้ำซ้อนเพราะเรามี Logo แล้ว
              headerSubtitle: "hidden",
              footer: "hidden" // ซ่อน Footer ของ Clerk ถ้าอยากใช้ของเดิม
            }
          }}
          routing="hash" // หรือ "path" ตามการตั้งค่าของคุณ
          signUpUrl="/register" 
        />

        {/* ส่วนท้ายเดิมของคุณ */}
        <div className="mt-8 text-center pt-4 border-t border-gray-100/60 flex flex-col items-center gap-1">
           <span className="text-[10px] uppercase tracking-widest text-gray-400 font-bold bg-gray-50 px-3 py-1.5 rounded-full border border-gray-100">
             Project CS-01: AI Powered Investment
           </span>
           <Link href="/admin" className="text-[11px] text-gray-400 hover:text-blue-600 hover:underline transition-colors mt-1.5">
             สำหรับผู้ดูแลระบบ (Admin)
           </Link>
        </div>
      </div>
    </div>
  );
}